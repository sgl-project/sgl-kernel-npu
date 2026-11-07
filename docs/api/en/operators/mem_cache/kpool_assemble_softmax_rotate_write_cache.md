# kpool_assemble_softmax_rotate_write_cache

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Verified |
| `torch_native` | Supported | Verified |

## Signature

```python
from typing import Optional
import torch
from sgl_kernel_npu.operators.mem_cache import kpool_assemble_softmax_rotate_write_cache

def kpool_assemble_softmax_rotate_write_cache(
    index_k_cache: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    n_from_tail: torch.Tensor,
    chunk_src_start: torch.Tensor,
    tail_logical_base: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: Optional[torch.Tensor] = None,
    *,
    num_programs: Optional[int] = None,
    impl="auto",
) -> None:
    ...
```

`auto` selects `triton`; omitting impl has the same effect. There is no runtime backend fallback.

## Description

Assemble each pool from a prefix in the request's tail ring and a suffix in the current chunk, according to the supplied plan. Apply channel-wise softmax pooling to keys using their scores plus positional scores, then write the BF16 result to the specified cache row.

Pooling uses FP32 arithmetic. The write mask can leave selected cache rows unchanged; tail and chunk inputs remain read-only. This operation stores pooled keys without rotation or quantization.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `index_k_cache` | `torch.Tensor` | Contiguous storage flattenable to `[cache_rows, 128]` | `torch.bfloat16` | Required | Index cache updated in place. |
| `chunk_k` | `torch.Tensor` | `[N, 128]` | `torch.bfloat16` | Required | Current-chunk keys. |
| `chunk_score` | `torch.Tensor` | Same as `chunk_k` | `torch.float16`, `torch.bfloat16`, or `torch.float32` | Required | Scores paired with `chunk_k`. |
| `tail_k` | `torch.Tensor` | `[R, T, 128]` | `torch.bfloat16` | Required | Request-local key rings; read only. |
| `tail_score` | `torch.Tensor` | Same as `tail_k` | `torch.float16`, `torch.bfloat16`, or `torch.float32` | Required | Scores paired with `tail_k`. |
| `req_pool_idx` | `torch.Tensor` | `[P]` | `torch.int64` | Required | Request/ring row for each of the `P` pools. |
| `n_from_tail` | `torch.Tensor` | `[P]` | `torch.int32` or `torch.int64` | Required | Number of leading slots read from the tail ring. |
| `chunk_src_start` | `torch.Tensor` | `[P]` | `torch.int64` | Required | First chunk row used by each pool's suffix. |
| `tail_logical_base` | `torch.Tensor` | `[P]` | `torch.int32` or `torch.int64` | Required | Logical first tail slot; physical positions wrap by `T`. |
| `ape` | `torch.Tensor` | `[pool_size, 128]` | `torch.float32` | Required | Additive pooling-position scores; `pool_size > 0`. |
| `loc` | `torch.Tensor` | `[P]` | `torch.int64` | Required | Flattened cache rows to write. |
| `write_mask` | `torch.Tensor` or `None` | `[P]` or — | `torch.bool` | `None` | Optional per-pool write enable. |
| `num_programs` | `int` or `None` | — | — | `None` | Optional execution-parallelism hint; must be positive when supplied. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `auto`, `triton`, `torch_native` |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `None` | `NoneType` | — | — | Mutation scope is specified above. |

All tensors must be on the same NPU. Callers supply valid indices, sufficient storage, and initialized history, and prevent conflicting state writes; wrappers do not validate every value range. Require K>0, T>0, and 0<=n_from_tail<=K.

## triton backend

Channel width is 128. Cache, tails and plan fields must be contiguous; chunk keys/scores, APE, locations and masks may be made contiguous internally.

## torch_native backend

Uses independent gathers, FP32 softmax, and cache writes. Reads only selected source branches, including all-tail/empty-chunk plans. num_programs has no effect on this backend.

## Example

Requires sgl-kernel-npu, torch_npu and an NPU runtime. The example explicitly selects `torch_native`. `triton` and `auto` are also available; isolate mutable state when comparing backends.

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import kpool_assemble_softmax_rotate_write_cache
cache = torch.zeros(1, 128, device='npu', dtype=torch.bfloat16)
chunk_k = torch.randn(4, 128, device='npu', dtype=torch.bfloat16)
chunk_score = torch.zeros(4, 128, device='npu', dtype=torch.float32)
tail_k = torch.randn(1, 4, 128, device='npu', dtype=torch.bfloat16)
tail_score = torch.zeros(1, 4, 128, device='npu', dtype=torch.float32)
ape = torch.zeros(4, 128, device='npu', dtype=torch.float32)
kpool_assemble_softmax_rotate_write_cache(
    cache,
    chunk_k,
    chunk_score,
    tail_k,
    tail_score,
    req_pool_idx=torch.tensor([0], device='npu', dtype=torch.int64),
    n_from_tail=torch.tensor([2], device='npu', dtype=torch.int32),
    chunk_src_start=torch.tensor([0], device='npu', dtype=torch.int64),
    tail_logical_base=torch.tensor([0], device='npu', dtype=torch.int32),
    ape=ape,
    loc=torch.tensor([0], device='npu', dtype=torch.int64),
    impl='torch_native',
)
```

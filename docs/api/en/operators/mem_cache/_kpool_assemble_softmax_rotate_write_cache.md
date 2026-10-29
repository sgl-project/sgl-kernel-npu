# kpool_assemble_softmax_rotate_write_cache

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Unverified |

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

Let `K=ape.shape[0]` and `T=tail_k.shape[1]`. For plan row p, the first `n_from_tail[p]` entries read tail slots `(tail_logical_base[p]+j)%T`; remaining entries read chunk rows `chunk_src_start[p]+j-n_from_tail[p]`. Writes the reduction to `loc[p]` unless `write_mask[p]=False`; an omitted mask enables all writes. Valid all-tail plans may use empty chunks. Only the cache changes; tails and chunks are read only.

Channel-wise softmax pooling uses FP32. Let `a[j,d]=score[j,d]+ape[j,d]` and `m[d]=max_j a[j,d]`:

$$z_d=\operatorname{BF16}\left(\frac{\sum_j e^{a_{j,d}-m_d}k_{j,d}}{\sum_j e^{a_{j,d}-m_d}}\right).$$

There is no Hadamard transform, FP8 quantization, or per-slot scale output.

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
| `num_programs` | `int` or `None` | — | — | `None` | Keyword-only Triton grid size. Must be positive when supplied; it is capped at `P`. By default, the backend uses the available Vector Core count capped at `P`. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `auto`, `triton`, `torch_native` |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `None` | `NoneType` | — | — | Mutation scope is specified above. |

All tensors must be on the same NPU. Callers supply valid indices, sufficient storage, and initialized history, and prevent conflicting state writes; wrappers do not validate every value range. Require K>0, T>0, and 0<=n_from_tail<=K.

## triton backend

Channel width is 128. Uses online stable softmax and a grid-stride loop. Cache, tails, and plan fields must be contiguous; chunk keys/scores, APE, loc, and mask may be normalized internally. Explicit positive num_programs is capped at P; the default uses Vector Core count, with an implementation limit if properties are unavailable.

## torch_native backend

Uses independent gathers, FP32 softmax, and cache writes. Reads only selected source branches, including all-tail/empty-chunk plans. num_programs has no effect on this backend.

Importing the public registry requires NPU architecture detection, including for `torch_native`. Support is limited to accepted A3 cases; architecture registration does not establish coverage of every chip, shape or dtype.

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

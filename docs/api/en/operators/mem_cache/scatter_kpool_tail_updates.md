# scatter_kpool_tail_updates

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Verified |
| `torch_native` | Supported | Verified |

## Signature

```python
import torch
from sgl_kernel_npu.operators.mem_cache import scatter_kpool_tail_updates

def scatter_kpool_tail_updates(
    pool,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    dst_logical_start: torch.Tensor,
    chunk_src_start: torch.Tensor,
    n_write: torch.Tensor,
    *,
    impl="auto",
) -> None:
    ...
```

`auto` selects `triton`; omitting impl has the same effect. There is no runtime backend fallback.

## Description

Copy selected consecutive rows of chunk keys and scores into request-local tail rings. Each plan entry specifies the request, source start, destination start and number of rows to copy; destination slots wrap around the ring.

Only the selected tail slots are modified. This operation copies values directly without pooling or quantization, and callers must avoid conflicting writes.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `pool` | Object | — | Must expose an integer `index_kpool` attribute | Required | Pool configuration. The wrapper reads `pool.index_kpool` as `pool_size`. |
| `chunk_k` | `torch.Tensor` | `[N, 128]` | Floating point; no explicit wrapper check | Required | Source keys; use the same dtype as `tail_k`. |
| `chunk_score` | `torch.Tensor` | Same as `chunk_k` | Floating point; no explicit wrapper check | Required | Source scores; use the same dtype as `tail_score`. |
| `tail_k` | `torch.Tensor` | `[R, T, 128]` | Floating point; must be compatible with `chunk_k` | Required | Destination key rings; updated in place. |
| `tail_score` | `torch.Tensor` | Same as `tail_k` | Floating point; must be compatible with `chunk_score` | Required | Destination score rings; updated in place. |
| `req_pool_idx` | `torch.Tensor` | `[P]` | Integer; no explicit wrapper check | Required | Destination request/ring row for each plan row. |
| `dst_logical_start` | `torch.Tensor` | `[P]` | Integer; no explicit wrapper check | Required | First logical destination slot for each plan row. |
| `chunk_src_start` | `torch.Tensor` | `[P]` | Integer; no explicit wrapper check | Required | First source chunk row for each plan row. |
| `n_write` | `torch.Tensor` | `[P]` | Integer; no explicit wrapper check | Required | Number of rows to copy for each plan row. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `auto`, `triton`, `torch_native` |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `None` | `NoneType` | — | — | Mutation scope is specified above. |

All tensors must be on the same NPU. Callers supply valid indices, sufficient storage, and initialized history, and prevent conflicting state writes; wrappers do not validate every value range.

## triton backend

Channel width is 128. Tails and one-dimensional plans must be contiguous; chunk keys/scores are made contiguous internally. Empty plans return immediately.

## torch_native backend

Uses independent PyTorch indexing and boolean masks; source/destination pairs should use compatible dtypes to avoid implicit conversion.

## Example

Requires sgl-kernel-npu, torch_npu and an NPU runtime. The example explicitly selects `torch_native`. `triton` and `auto` are also available; isolate mutable state when comparing backends.

```python
from types import SimpleNamespace
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import scatter_kpool_tail_updates
pool = SimpleNamespace(index_kpool=2)
chunk_k = torch.randn(2, 128, device='npu', dtype=torch.bfloat16)
chunk_score = torch.zeros(2, 128, device='npu', dtype=torch.bfloat16)
tail_k = torch.zeros(1, 4, 128, device='npu', dtype=torch.bfloat16)
tail_score = torch.zeros(1, 4, 128, device='npu', dtype=torch.bfloat16)
scatter_kpool_tail_updates(
    pool,
    chunk_k,
    chunk_score,
    tail_k,
    tail_score,
    req_pool_idx=torch.tensor([0], device='npu', dtype=torch.int32),
    dst_logical_start=torch.tensor([0], device='npu', dtype=torch.int32),
    chunk_src_start=torch.tensor([0], device='npu', dtype=torch.int32),
    n_write=torch.tensor([2], device='npu', dtype=torch.int32),
    impl='torch_native',
)
```

# scatter_kpool_tail_updates_npu

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |

## Signature

```python
from sgl_kernel_npu.operators.mem_cache import scatter_kpool_tail_updates_npu

def scatter_kpool_tail_updates_npu(
    pool,
    chunk_k,
    chunk_score,
    tail_k,
    tail_score,
    req_pool_idx,
    dst_logical_start,
    chunk_src_start,
    n_write,
    *,
    impl="auto",
):
    ...
```

## Description

Scatters key/score rows from a chunk into request-local tail rings. For plan row `r` and `j < n_write[r]`, the source is `chunk_src_start[r] + j`; the destination request is `req_pool_idx[r]`, and its physical tail slot is `(dst_logical_start[r] + j) % T`.

## Parameters

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
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"` or `"triton"`. |

## Return value and in-place state

Returns `None`; only `tail_k` and `tail_score` are modified. `P = req_pool_idx.shape[0]`, and `P == 0` is a no-op. `pool`, the chunk tensors, and all plan tensors are read only.

The wrapper does not comprehensively validate shapes, dtypes, devices, or ranges. Callers must provide non-negative write counts and valid source/request/slot ranges. `chunk_k` and `chunk_score` are made contiguous internally; mutable tail buffers must have the layout expected by the kernel, including a contiguous channel dimension, and the one-dimensional plan tensors must be contiguous. All tensors must be on the same NPU.

## triton backend

- Uses the fixed 128-channel layout and a grid-stride loop over plan rows. It copies native elements without a BF16 conversion; use matching key dtypes and matching score dtypes. The backend does not expose `num_programs` for this operator.
- `auto` aliases `triton`, not a dynamic choice or fallback. Registry import still performs ACL architecture detection.

## Example

```python
from types import SimpleNamespace

import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import scatter_kpool_tail_updates_npu

pool = SimpleNamespace(index_kpool=2)
chunk_k = torch.randn(2, 128, device="npu", dtype=torch.bfloat16)
chunk_score = torch.zeros(2, 128, device="npu", dtype=torch.bfloat16)
tail_k = torch.zeros(1, 4, 128, device="npu", dtype=torch.bfloat16)
tail_score = torch.zeros(1, 4, 128, device="npu", dtype=torch.bfloat16)

scatter_kpool_tail_updates_npu(
    pool,
    chunk_k,
    chunk_score,
    tail_k,
    tail_score,
    req_pool_idx=torch.tensor([0], device="npu", dtype=torch.int32),
    dst_logical_start=torch.tensor([0], device="npu", dtype=torch.int32),
    chunk_src_start=torch.tensor([0], device="npu", dtype=torch.int32),
    n_write=torch.tensor([2], device="npu", dtype=torch.int32),
)
```

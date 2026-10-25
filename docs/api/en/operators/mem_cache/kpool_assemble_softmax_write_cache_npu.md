# kpool_assemble_softmax_write_cache_npu

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |

## Signature

```python
from sgl_kernel_npu.operators.mem_cache import (
    kpool_assemble_softmax_write_cache_npu,
)

def kpool_assemble_softmax_write_cache_npu(
    index_k_cache,
    chunk_k,
    chunk_score,
    tail_k,
    tail_score,
    req_pool_idx,
    n_from_tail,
    chunk_src_start,
    tail_logical_base,
    ape,
    loc,
    write_mask=None,
    *,
    num_programs=None,
    impl="auto",
):
    ...
```

## Description

Assembles one pooled BF16 key for each plan row. The first `n_from_tail[row]` slots come from the request-local tail ring, starting at `tail_logical_base[row]` and wrapping by `T`; the remaining slots come from `chunk_k`, starting at `chunk_src_start[row]`.

For each channel, the kernel applies softmax to `score + ape` over the pool slots, computes the weighted key, converts it to BF16, and writes it to flattened cache row `loc[row]`. A false `write_mask[row]` leaves that cache row unchanged.

## Parameters

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
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"` or `"triton"`. |

## Return value and in-place state

Returns `None`; only `index_k_cache` is modified. `P = req_pool_idx.numel()`, and `P == 0` is a no-op. All plan tensors must have `P` entries. The source ranges and the per-pool slot counts must be valid upstream.

All tensors must be on the same NPU, and the final kernel inputs must be contiguous. The wrapper makes `chunk_k`, `chunk_score`, `ape`, `loc`, and optional `write_mask` contiguous internally. `index_k_cache`, both tail tensors, and the other plan tensors must already satisfy the contiguity assertions.

## triton backend

- Uses a fixed channel width of 128 and FP32 pooling accumulation; the result is stored as BF16 without Hadamard or FP8 processing.
- `auto` aliases `triton`; there is no dynamic selection or fallback. Registry import still requires NPU/ACL architecture detection.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import (
    kpool_assemble_softmax_write_cache_npu,
)

cache = torch.zeros(1, 128, device="npu", dtype=torch.bfloat16)
chunk_k = torch.randn(4, 128, device="npu", dtype=torch.bfloat16)
chunk_score = torch.zeros(4, 128, device="npu", dtype=torch.float32)
tail_k = torch.randn(1, 4, 128, device="npu", dtype=torch.bfloat16)
tail_score = torch.zeros(1, 4, 128, device="npu", dtype=torch.float32)
ape = torch.zeros(4, 128, device="npu", dtype=torch.float32)

kpool_assemble_softmax_write_cache_npu(
    cache,
    chunk_k,
    chunk_score,
    tail_k,
    tail_score,
    req_pool_idx=torch.tensor([0], device="npu", dtype=torch.int64),
    n_from_tail=torch.tensor([2], device="npu", dtype=torch.int32),
    chunk_src_start=torch.tensor([0], device="npu", dtype=torch.int64),
    tail_logical_base=torch.tensor([0], device="npu", dtype=torch.int32),
    ape=ape,
    loc=torch.tensor([0], device="npu", dtype=torch.int64),
)
```

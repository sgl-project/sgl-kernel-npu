# kpool_decode_update_and_maybe_write_cache

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Verified |
| `torch_native` | Supported | Verified |

## Signature

```python
import torch
from sgl_kernel_npu.operators.mem_cache import kpool_decode_update_and_maybe_write_cache

def kpool_decode_update_and_maybe_write_cache(
    buf: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    block_tables: torch.Tensor,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    *,
    pool_size: int,
    slots_per_page: int,
    head_dim: int = 128,
    impl="auto",
) -> None:
    ...
```

`auto` selects `triton`; omitting impl has the same effect. There is no runtime backend fallback.

## Description

Write the current decode key and score into the request's tail ring. When this token completes a pool, combine it with the stored tail history, apply channel-wise softmax pooling, and write the BF16 result to the pooled cache using the page table.

Pooling uses FP32 arithmetic and includes positional scores. Invalid request rows leave all state unchanged. `out_cache_loc` controls whether the row is valid; the pooled-cache destination is resolved separately from the page table.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `buf` | `torch.Tensor` | `[num_pages, S*D]` / `[num_pages, S, 1, D]` | `torch.bfloat16` | Required | In-place cache; S=slots_per_page, D=head_dim, page stride must equal S*D. |
| `tail_k` | `torch.Tensor` | `[R, T, head_dim]` | `torch.bfloat16` | Required | Request-local key rings; updated in place for valid rows. |
| `tail_score` | `torch.Tensor` | Same as `tail_k` | `torch.bfloat16` / `torch.float32` | Required | Paired scores; current tests cover BF16 and FP32, with FP32 reduction. |
| `key` | `torch.Tensor` | `[B, head_dim]` | `torch.bfloat16` | Required | Current decode keys. |
| `slot_score` | `torch.Tensor` | `[B, head_dim]` | `torch.bfloat16` / `torch.float32` | Required | Paired scores; current tests cover BF16 and FP32, with FP32 reduction. |
| `ape` | `torch.Tensor` | `[pool_size, head_dim]` | `torch.float32` | Required | Additive pooling-position scores. |
| `block_tables` | `torch.Tensor` | `[B, BLOCK_TABLE_COLS]` | `torch.int32` | Required | Token-page table used to locate a completed pool's cache page. |
| `req_pool_indices` | `torch.Tensor` | `[>= B]` | Integer; not explicitly checked by the wrapper | Required | Request/ring row for each batch item. |
| `positions` | `torch.Tensor` | `[>= B]` | Integer; not explicitly checked by the wrapper | Required | Current token position for each batch item. |
| `seq_lens` | `torch.Tensor` | `[>= B]` | Integer; not explicitly checked by the wrapper | Required | Sequence length for each batch item. |
| `out_cache_loc` | `torch.Tensor` | `[>= B]` | Integer; not explicitly checked by the wrapper | Required | Token cache locations; a zero entry suppresses that row's updates. Read only. |
| `pool_size` | `int` | — | — | Required | Keyword-only number of tokens in one pool. |
| `slots_per_page` | `int` | — | — | Required | Keyword-only number of pooled slots per flattened cache page. |
| `head_dim` | `int` | — | — | `128` | Keyword-only channel width; must match the tail's last dimension. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `auto`, `triton`, `torch_native` |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `None` | `NoneType` | — | — | Mutation scope is specified above. |

All tensors must be on the same NPU. Callers supply valid indices, sufficient storage, and initialized history, and prevent conflicting state writes; wrappers do not validate every value range. Require positive pool_size, T, and slots_per_page, and a nonempty page table; only buf, tail_k, and tail_score are mutated.

## triton backend

buf, tail_k and tail_score must be contiguous; noncontiguous mutable state is rejected. Read-only inputs are normalized internally. buf.stride(0) must equal slots_per_page*head_dim; compatible 2D/4D layouts are allowed. Empty batches return immediately.

## torch_native backend

Uses independent row-wise PyTorch processing, stable FP32 softmax, and page-table writes. It does not duplicate every Triton assertion; callers must satisfy the public layout contract.

## Example

Requires sgl-kernel-npu, torch_npu and an NPU runtime. The example explicitly selects `torch_native`. `triton` and `auto` are also available; isolate mutable state when comparing backends.

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import kpool_decode_update_and_maybe_write_cache
b, r, t, h = (1, 1, 4, 128)
buf = torch.zeros(1, 8 * h, device='npu', dtype=torch.bfloat16)
tail_k = torch.zeros(r, t, h, device='npu', dtype=torch.bfloat16)
tail_score = torch.zeros(r, t, h, device='npu', dtype=torch.float32)
key = torch.randn(b, h, device='npu', dtype=torch.bfloat16)
slot_score = torch.zeros(b, h, device='npu', dtype=torch.float32)
ape = torch.zeros(4, h, device='npu', dtype=torch.float32)
block_tables = torch.zeros(b, 1, device='npu', dtype=torch.int32)
metadata = dict(
    req_pool_indices=torch.zeros(b, device='npu', dtype=torch.int32),
    positions=torch.tensor([3], device='npu', dtype=torch.int32),
    seq_lens=torch.tensor([4], device='npu', dtype=torch.int32),
    out_cache_loc=torch.ones(b, device='npu', dtype=torch.int32),
)
kpool_decode_update_and_maybe_write_cache(
    buf,
    tail_k,
    tail_score,
    key,
    slot_score,
    ape,
    block_tables,
    **metadata,
    pool_size=4,
    slots_per_page=8,
    impl='torch_native',
)
```

# append_local_block_to_topk_idx

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

Only the `triton` and `torch_native` implementations exist; both are supported on arch22 and arch35.

## Signature

```python
from sgl_kernel_npu.operators.indexer import append_local_block_to_topk_idx

def append_local_block_to_topk_idx(
    topk_idx,
    seq_lens,
    block_size,
    num_blocks,
    *,
    impl="auto",
):
    ...
```

`auto` and `triton` currently refer to the same entry point; execution failures do not trigger
an automatic fallback. `torch_native` is an independent PyTorch reference used for accuracy
comparison.

## Description

In short: append the block id that the current query belongs to (the local block) after the topk
blocks already selected by sparse attention.

`topk_idx` holds the block ids chosen by sparse attention, with shape `[N, B, topk]`. The operator
adds one more entry per row, the block the query belongs to, and returns `[N, B, topk + 1]`. For
each `(n, b)`, let `seq_len = seq_lens[b]`:

1. The query position is `q_pos = max(seq_len - 1, 0)`.
2. The block the query belongs to is `local = min(q_pos // block_size, num_blocks - 1)`;
   `num_blocks` is the exclusive block-id upper bound, so larger values clamp to the last
   block. Require `num_blocks >= ceil(max_seq_len / block_size)`; otherwise `local` is clamped
   too low.
3. Filter the input candidates: keep a block id `c` only when `0 <= c < num_blocks` and
   `c * block_size <= q_pos`; otherwise write `-1`. This drops out-of-range blocks and blocks
   placed after the query.
4. Put `local` into the last column: `-1` if the filtered candidates already contain it (avoid a
   duplicate), otherwise `local`.

Notes:

- The first `topk` columns keep the input order; duplicate candidates are kept as-is and only the
  appended `local` is deduplicated.
- `num_blocks` is the exclusive block-id upper bound, not page-table capacity and not the per-row
  block count. SGLang passes `max_blocks = ceil(max_seqlen / block_size)` derived from `max_seqlen`.
- With `seq_len = 0`, `q_pos = 0` and `local = 0`; only block 0 passes the filter.
- `topk` must be greater than zero.
- The input is not modified; a new tensor is returned.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `topk_idx` | `torch.Tensor` | `[N, B, topk]` | `torch.int32` | Required | Candidate block ids; `N` is the head count (`num_kv_heads` at block-sparse attention call sites). Must be contiguous; Triton requires the NPU while `torch_native` is device-independent. `N > 0`, `B > 0`, `topk > 0`. |
| `seq_lens` | `torch.Tensor` | `[B]` | `torch.int32`, `torch.int64` | Required | `seq_len` per row: how many KV tokens the query can see, including itself; values should be in `[0, INT32_MAX]`. Must be contiguous, on the same device as `topk_idx`, with length exactly `B`. |
| `block_size` | `int` | — | — | Required | Tokens per KV block; must be positive. |
| `num_blocks` | `int` | — | — | Required | Exclusive upper bound of logical block ids; must be positive and at least `ceil(max_seq_len / block_size)`. Not the per-row block count and not page-table capacity. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"triton"` or `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[N, B, topk + 1]` | `torch.int32` | Newly allocated output on the input device. The input is not modified. |

The entry point validates nine assertions: `topk_idx` being 3-D, `int32`, contiguous and
`topk > 0`; `seq_lens` being 1-D with length `B` and contiguous; positive `block_size` and
`num_blocks`. Violations raise `AssertionError`; there is no fallback to another implementation.
The `seq_lens` dtype (`int32`/`int64`), device consistency and a non-empty batch are caller
requirements that the entry point does not validate.

## triton backend

- Uses `_append_local_block_to_topk_idx_kernel` with `BSQ = 16` rows per program, a grid of
  `(cdiv(B, 16), N)`, `BLOCK_SIZE_T` injected from `triton.next_power_of_2(topk)` via heuristics,
  `num_warps = 4` and `num_stages = 1`.
- Reads candidates through the provided strides and writes a contiguous output; inputs must be
  contiguous.
- The candidate check `candidate * block_size` is int32 arithmetic (`block_size` is constexpr);
  it can overflow when `(num_blocks - 1) * block_size >= 2^31` (for `block_size = 128` that needs
  `num_blocks >= 16,777,217`), which is outside the supported range.
- `seq_len` is read as int32; values outside `[0, INT32_MAX]` are truncated and unsupported.

## torch_native backend

- Fully vectorized PyTorch implementation. Validation arithmetic uses int64 and the same input
  constraints apply; the input is likewise not modified.
- Differences from Triton only appear on out-of-range inputs: candidate products overflowing
  int32, and `seq_len` outside the int32 range. Within the supported range both are bitwise
  identical.

## Example

Install a wheel containing this API and the NPU runtime dependencies before running the example.

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.indexer import append_local_block_to_topk_idx

# seq_len = 513 -> q_pos = 512, local block = min(512 // 128, 7) = 4.
topk_idx = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.int32, device="npu")
seq_lens = torch.tensor([513], dtype=torch.int32, device="npu")

print(list_impls(append_local_block_to_topk_idx))  # ['auto', 'torch_native', 'triton']
y_default = append_local_block_to_topk_idx(topk_idx, seq_lens, 128, 8)
y_native = append_local_block_to_topk_idx(topk_idx, seq_lens, 128, 8, impl="torch_native")
print(y_default.tolist())  # [[[0, 1, 2, 3, 4]]]
torch.testing.assert_close(y_default, y_native, rtol=0, atol=0)
```

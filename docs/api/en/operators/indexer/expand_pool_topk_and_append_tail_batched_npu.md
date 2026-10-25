# expand_pool_topk_and_append_tail_batched_npu

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |

## Signature

The public name has no leading underscore; the backend implementation is named `_expand_pool_topk_and_append_tail_batched_npu`.

```python
from sgl_kernel_npu.operators.indexer import (
    expand_pool_topk_and_append_tail_batched_npu,
)

def expand_pool_topk_and_append_tail_batched_npu(
    pool_indices,
    seq_lens_per_token,
    n_real,
    num_q_padded,
    pool_size,
    index_topk,
    *,
    impl="auto",
):
    ...
```

## Description

Expands each pool-level top-k index `p` into the token indices `p * pool_size, ..., p * pool_size + pool_size - 1`. It then appends the uncompressed tail for each valid query row.

For `seq_len = seq_lens_per_token[row]`, the tail starts at `(seq_len // pool_size) * pool_size` and contains `seq_len % pool_size` indices. Unused tail slots and output rows beyond the valid row count are filled with `-1`.

## Parameters

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `pool_indices` | `torch.Tensor` | `[input_rows, N_POOL_TOPK]`, with enough rows for the effective `n_real` | `torch.int32` | Required | Selected pool positions. Row strides are passed to the kernel. |
| `seq_lens_per_token` | `torch.Tensor` | `[input_rows]`, with enough rows for the effective `n_real` | `torch.int32` | Required | Request length for each query row; use contiguous one-dimensional storage. |
| `n_real` | `int` | — | — | Required | Requested valid row count. The backend clamps it to the smaller input row count before launching. |
| `num_q_padded` | `int` | — | — | Required | Output row count; must be at least the effective `n_real`. |
| `pool_size` | `int` | — | — | Required | Number of tokens represented by one pool; use a valid positive pooling width. |
| `index_topk` | `int` | — | — | Required | Token-level top-k; must equal `N_POOL_TOPK * pool_size`. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"` or `"triton"`. |

## Return value

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[num_q_padded, index_topk + pool_size - 1]` | `torch.int32` | Expanded indices followed by tail slots. Inputs are not modified. |

The backend allocates physical rows padded to a multiple of eight int32 columns and returns a slice with the logical column count. Consequently, callers must not assume that the returned view is contiguous.

## triton backend

- Checks two-dimensional/int32 `pool_indices`, one-dimensional/int32 `seq_lens_per_token`, and `index_topk == N_POOL_TOPK * pool_size`. It uses `BLOCK_M = 4` and masks rows beyond the effective `n_real`.
- `auto` aliases `triton`; no dynamic selection or fallback is performed. Registry import still performs ACL architecture detection.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.indexer import (
    expand_pool_topk_and_append_tail_batched_npu,
)

pool_indices = torch.tensor(
    [[0, 1], [0, 0]], device="npu", dtype=torch.int32
)
seq_lens = torch.tensor([10, 0], device="npu", dtype=torch.int32)
out = expand_pool_topk_and_append_tail_batched_npu(
    pool_indices,
    seq_lens,
    n_real=1,
    num_q_padded=2,
    pool_size=4,
    index_topk=8,
)

print(out.shape)  # torch.Size([2, 11]); row 1 is padding with -1
```

# compute_n_gram_ids

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `auto` / `ascendc` | Supported implementation | Unsupported |
| `torch_native` | Independent reference | Unsupported |

See the [build and installation guide](../../../../../python/sgl_kernel_npu/README.md).
This table describes implementation scope, not a new hardware validation result.

## Signature

```python
from sgl_kernel_npu.operators.embedding import compute_n_gram_ids

def compute_n_gram_ids(
    oe_weights, oe_mods, exclusive_oe_embeder_size_sums,
    tokens, exclusive_req_len_sums, oe_token_table, row_indices, column_starts,
    batch_size, oe_n, oe_k, max_context_len, *, impl="auto",
):
    ...
```

`auto` selects `ascendc`, which calls `torch.ops.npu.compute_n_gram_ids`.
There is no automatic fallback or dtype conversion.

## Description

Compute LongCat2 over-embedding IDs from the **already updated** token table.
`tokens` supplies the output token count; its values are not substituted for
history. Call [update_oe_token_table](update_oe_token_table.md) first when writing
new tokens.

Let `C[r] = sum(req_lens[:r+1])` and `C[-1] = 0`. Despite its historical name,
`exclusive_req_len_sums` stores these **inclusive request ends**, has length `B`,
and has no leading zero. For packed token `i` in request `r`, read history from
column `p = column_starts[r] + i - C[r-1]` of row `row_indices[r]`.
For n-gram order `n = 2..oe_n` and embedding group `k = 0..oe_k-1`:

```text
f = (n - 2) * oe_k + k
id[i, f] = offset[f] + (sum_j ((history[p-j] * weight[n-2,k,j]) % mod[n-2,k])) % mod[n-2,k]
```

Visit `j = 0..n-1`, stopping before crossing the request row's beginning or at
the first negative history token. A negative token is a boundary sentinel, not a
token to skip while continuing farther back. Empty sums produce the group offset.

## Parameters and return value

All Tensor inputs follow the existing contiguous, same-NPU caller convention.
The table documents that convention; the public wrapper only dispatches.
Write ranges and packed request lengths are supplied consistently by the caller.

| Parameter | Type / shape / dtype | Default | Meaning |
| --- | --- | --- | --- |
| `oe_weights` | Tensor `[oe_n-1, oe_k, oe_n]`, int32 | Required | Modular positional weights; existing model tables use nonnegative weights. |
| `oe_mods` | Tensor `[oe_n-1, oe_k]`, int32 | Required | Positive modulus for each group. |
| `exclusive_oe_embeder_size_sums` | Tensor `[(oe_n-1)*oe_k+1]`, int32 | Required | Embedding-table offsets with leading zero and final total; each output uses entry `f`. |
| `tokens` | Tensor `[T]`, int32 | Required | Packed current tokens; determines output row count. |
| `exclusive_req_len_sums` | Tensor `[B]`, int32 | Required | Inclusive request ends; final entry is `T`. |
| `oe_token_table` | Tensor `[R, max_context_len]`, int32 | Required | Updated history; negative entries terminate an n-gram. The native row stride is `max_context_len`. |
| `row_indices` | Tensor `[B]`, int64 | Required | History row for each request. |
| `column_starts` | Tensor `[B]`, int32 | Required | First current-token column in each history row. |
| `batch_size` | int | Required | `B > 0`. |
| `oe_n` | int | Required | Maximum n-gram order, greater than 1. |
| `oe_k` | int | Required | Number of groups per order, greater than 0. |
| `max_context_len` | int | Required | Positive history row width used by native addressing. |
| `impl` | str | `"auto"` | Keyword-only: `auto`, `ascendc`, `torch_native`. |

| Return | Type / shape / dtype | Meaning |
| --- | --- | --- |
| IDs | Tensor `[T, (oe_n-1)*oe_k]`, int32 | New same-NPU tensor, ordered by n-gram order then group. Inputs are not modified. |

## ascendc backend

Uses the official standalone CANN host, tiling, and kernel through this
repository's Torch adapter, calling `aclnnSglComputeNGramIdsGetWorkspaceSize` and
`aclnnSglComputeNGramIds`. CANN manages tiling and kernel dispatch; the handwritten
tiling cache from the old direct-launch path is no longer used. The public API
adds no input normalization or new input checks. Use the root REFACTOR build;
`--ops all` includes this operator on arch22.

## torch_native backend

Independently gathers history windows and calculates their weighted sums using
PyTorch. It is a correctness reference and preserves the int32 output and existing
sentinel behavior.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.embedding import compute_n_gram_ids

device = "npu:0"
torch.npu.set_device(device)
weights = torch.tensor([[[1, 2]]], device=device, dtype=torch.int32)
mods = torch.tensor([[7]], device=device, dtype=torch.int32)
offsets = torch.tensor([0, 7], device=device, dtype=torch.int32)
tokens = torch.tensor([2, 3], device=device, dtype=torch.int32)
ends = torch.tensor([2], device=device, dtype=torch.int32)
history = torch.tensor([[2, 3, -1, -1]], device=device, dtype=torch.int32)
rows = torch.tensor([0], device=device, dtype=torch.int64)
starts = torch.tensor([0], device=device, dtype=torch.int32)
ids = compute_n_gram_ids(
    weights, mods, offsets, tokens, ends, history, rows, starts,
    batch_size=1, oe_n=2, oe_k=1, max_context_len=4, impl="ascendc",
)
# Mathematical result: [[2], [0]]. Use impl="torch_native" for the reference.
```

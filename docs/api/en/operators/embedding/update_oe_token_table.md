# update_oe_token_table

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `auto` / `ascendc` | Supported implementation | Unsupported |
| `torch_native` | Independent reference | Unsupported |

See the [build and installation guide](../../../../../python/sgl_kernel_npu/README.md).
This table describes implementation scope, not a new hardware validation result.

## Signature

```python
from sgl_kernel_npu.operators.embedding import update_oe_token_table

def update_oe_token_table(
    tokens, req_lens, row_indices, column_starts, ignore_tokens,
    batch_size, max_context_len, oe_token_table, *, impl="auto",
):
    ...
```

`auto` selects `ascendc`, calling `torch.ops.npu.update_oe_token_table`.
There is no automatic fallback or dtype conversion.

## Description

Write packed current tokens into the LongCat2 over-embedding history **in place**.
For request `r`, let `s_r = sum(req_lens[:r])`. For `0 <= j < req_lens[r]`:

```text
x = tokens[s_r + j]
oe_token_table[row_indices[r], column_starts[r] + j] = -1 if x in ignore_tokens else x
```

Only the current write ranges change. Existing history outside those ranges is
preserved, even if it contains an ignored token ID. Empty `ignore_tokens` means
no replacement. A request of length zero performs no writes.

The negative sentinel is consumed by [compute_n_gram_ids](compute_n_gram_ids.md)
as a history boundary. Call this update before computing IDs for the current tokens.

## Parameters and return value

All Tensor inputs follow the existing contiguous, same-NPU caller convention.
Use the history row width as `max_context_len`; the native kernel addresses rows
with that stride. The caller supplies valid, non-conflicting request write ranges
and lengths matching the packed tokens. These are existing calling conventions,
not additional checks in the public wrapper.

| Parameter | Type / shape / dtype | Default | Meaning |
| --- | --- | --- | --- |
| `tokens` | Tensor `[T]`, int32 | Required | Packed current tokens, where `T = sum(req_lens)`. |
| `req_lens` | Tensor `[B]`, int32 | Required | Per-request token counts, not cumulative ends. |
| `row_indices` | Tensor `[B]`, int64 | Required | Destination history row for each request. |
| `column_starts` | Tensor `[B]`, int32 | Required | Start column for each write. |
| `ignore_tokens` | Tensor `[I]`, int32 | Required | Token IDs replaced by `-1` in the current write. `I=0` is allowed. |
| `batch_size` | int | Required | `B > 0`. |
| `max_context_len` | int | Required | Positive history row width used by native addressing. |
| `oe_token_table` | Tensor `[R, max_context_len]`, int32 | Required | Mutable history table. |
| `impl` | str | `"auto"` | Keyword-only: `auto`, `ascendc`, `torch_native`. |

| Return | Type / shape / dtype | Meaning |
| --- | --- | --- |
| Updated table | Tensor `[R, max_context_len]`, int32 | Aliases the input `oe_token_table` on the same NPU; no replacement table is returned. |

## ascendc backend

Uses the official standalone CANN host, tiling, and kernel through this
repository's Torch adapter, calling `aclnnSglUpdateOeTokenTableGetWorkspaceSize` and
`aclnnSglUpdateOeTokenTable`. The original table is passed as the ACLNN output and
returned by the adapter; its Torch schema marks `oe_token_table` and the return
value with the same `Tensor(a!)` alias. CANN manages tiling and kernel
dispatch; the old handwritten tiling cache is no longer used. Use the root
REFACTOR build; `--ops all` includes this operator on arch22. The public wrapper
adds no input normalization or validation.

## torch_native backend

Independently computes packed-token positions and ignore-token replacement with
PyTorch, then updates the input table. It retains the same alias and write-region
semantics. Accuracy comparisons must use separate initial tables for the two implementations.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.embedding import update_oe_token_table

device = "npu:0"
torch.npu.set_device(device)
table = torch.full((2, 8), -1, device=device, dtype=torch.int32)
tokens = torch.tensor([5, 9, 6], device=device, dtype=torch.int32)
lengths = torch.tensor([2, 1], device=device, dtype=torch.int32)
rows = torch.tensor([1, 0], device=device, dtype=torch.int64)
starts = torch.tensor([0, 2], device=device, dtype=torch.int32)
ignored = torch.tensor([9], device=device, dtype=torch.int32)
updated = update_oe_token_table(
    tokens, lengths, rows, starts, ignored, 2, 8, table, impl="ascendc",
)
assert updated.data_ptr() == table.data_ptr()
# Writes row 1 columns 0:2 as [5, -1], and row 0 column 2 as 6.
# Select impl="torch_native" to run the independent reference.
```

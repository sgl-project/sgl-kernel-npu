# build_out_tokens

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Unverified |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

The default `impl="auto"` selects the same implementation as `"triton"`.
Registration does not establish hardware validation; the table records the
currently verified status.

## Signature

```python
from sgl_kernel_npu.operators.sample import build_out_tokens

def build_out_tokens(
    *, draft_tokens, correct_len, bonus, verify_num_draft_tokens, gamma, impl="auto"
):
    ...
```

## Description

Creates a `[B, gamma + 1]` output buffer by copying each row of draft tokens,
appending a zero, and writing the bonus token at `correct_len` for that row.
The draft entries after the bonus are retained: use the acceptance length to
identify the committed prefix. `correct_len` must be in `[0, gamma]`.

## Parameters and return value

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `draft_tokens` | `torch.Tensor` | `[B, gamma]`, int32 or int64 | Required | Draft token IDs on NPU. |
| `correct_len` | `torch.Tensor` | `[B]`, int32 or int64 | Required | Accepted draft length for each row, on the same device. |
| `bonus` | `torch.Tensor` | `[B]`, int32 or int64 | Required | Bonus token ID for each row, on the same device. |
| `verify_num_draft_tokens` | `int` | - | Required | Must equal `gamma + 1`. |
| `gamma` | `int` | - | Required | Number of draft tokens per row; must be nonnegative. |
| `impl` | `str` | - | `"auto"` | `"auto"`, `"triton"`, or `"torch_native"`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[B, gamma + 1]`, `torch.int64` | Newly allocated buffer on the input device. Inputs are not modified. |

## triton backend

Accepts strided inputs and constructs the output with a Triton kernel. The
caller is responsible for ensuring each acceptance length is in range.

## torch_native backend

Uses Torch concatenation and scatter to construct the same output.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.sample import build_out_tokens

drafts = torch.tensor([[10, 11, 12], [20, 21, 22]], device="npu")
lengths = torch.tensor([1, 3], device="npu")
bonus = torch.tensor([99, 98], device="npu")
out = build_out_tokens(draft_tokens=drafts, correct_len=lengths, bonus=bonus,
                       verify_num_draft_tokens=4, gamma=3)
# out: [[10, 99, 12, 0], [20, 21, 22, 98]]
# Valid prefixes have lengths lengths + 1.
```

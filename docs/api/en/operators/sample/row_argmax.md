# row_argmax

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
from sgl_kernel_npu.operators.sample import row_argmax

def row_argmax(x, *, impl="auto"):
    ...
```

## Description

Returns the index of the maximum value in each row, matching
`torch.argmax(x, dim=-1)`. Ties select the lowest column index; if a row
contains NaNs, the first NaN is selected.

## Parameters and return value

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[B, V]`, `torch.float32` | Required | NPU input with `V > 0` and unit column stride. `B` may be zero; rows may have a larger stride. |
| `impl` | `str` | - | `"auto"` | `"auto"`, `"triton"`, or `"torch_native"`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `indices` | `torch.Tensor` | `[B]`, `torch.int64` | Newly allocated indices on the input device. |

## triton backend

Reduces each row using a tiled Triton kernel. The input must be two-dimensional,
FP32 and have unit column stride.

## torch_native backend

Uses `torch.argmax(x, dim=-1)`.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.sample import row_argmax

x = torch.tensor([[1.0, 3.0, 3.0], [4.0, 2.0, 1.0]], device="npu")
indices = row_argmax(x)
# indices: tensor([1, 0], device="npu:0")
```

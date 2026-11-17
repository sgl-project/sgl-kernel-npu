# div_argmax

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

The default `impl="auto"` selects the same implementation as `"triton"`.
Registration does not establish hardware validation; the table records the
currently verified status.

## Signature

```python
from sgl_kernel_npu.operators.sample import div_argmax

def div_argmax(x, divisor, *, divisor_min=None, ignore_nan=False, impl="auto"):
    ...
```

## Description

Returns the index of the maximum value of `(x / divisor)` per row, where the
division is element-wise. Computation rules:

- When `divisor_min` is not None, `divisor` is first clamped with
  `clamp_min(divisor_min)` before the division; NaN entries of `divisor`
  stay NaN.
- Ties select the lowest column index.
- With `ignore_nan=True`, a NaN quotient is treated as `-inf` and never wins;
  a row with nothing above `-inf` gives index 0.
- With `ignore_nan=False`, NaN quotients follow the default NaN behavior of
  Ascend `tl.max`, which is not guaranteed to match `torch.argmax` (an
  all-NaN tile leaves its index at the row sentinel `N`); do not rely on
  results in that case.

## Parameters and return value

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[B, V]`, `torch.float32` | Required | NPU input with `V >= 1` and unit column stride; rows may have a larger stride. |
| `divisor` | `torch.Tensor` | `[B, V]`, `torch.float32` | Required | Divisor with the same shape on the same NPU, unit column stride. |
| `divisor_min` | `float` | - | `None` | Lower bound for the divisor, clamped before the division. |
| `ignore_nan` | `bool` | - | `False` | Whether a NaN quotient is forced to `-inf`. |
| `impl` | `str` | - | `"auto"` | `"auto"`, `"triton"`, or `"torch_native"`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `indices` | `torch.Tensor` | `[B]`, `torch.int64` | Newly allocated indices on the input device; `B` may be zero. |

## triton backend

Two-stage reduction: the partial kernel splits each row into blocks
(block size 4096 for up to 256 rows, otherwise 8192) and computes each
block's maximum and index in parallel; the final kernel reduces the
per-block candidates. Ascend Triton cannot parse the two-operand `tl.reduce`
(it lowers to a region `tt.reduce`), so one-operand `tl.max`/`tl.min` are
used and a NaN quotient never wins. The input must be two-dimensional, FP32
and have unit column stride.

## torch_native backend

Runs `divisor.clamp_min(divisor_min)` (skipped for `None`), divides
element-wise, replaces NaN quotients with `-inf` when `ignore_nan` is true,
and finally calls `torch.argmax(dim=-1)`.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.sample import div_argmax

x = torch.tensor([[1.0, 6.0, 3.0], [4.0, 2.0, 1.0]], device="npu")
divisor = torch.tensor([[1.0, 2.0, 1.0], [2.0, 1.0, 1.0]], device="npu")
indices = div_argmax(x, divisor, divisor_min=1e-38, ignore_nan=True)
# x / divisor is [1.0, 3.0, 3.0] and [2.0, 2.0, 1.0]; indices: [1, 0]
```

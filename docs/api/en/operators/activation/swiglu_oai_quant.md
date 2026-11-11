# swiglu_oai_quant

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.activation import swiglu_oai_quant

def swiglu_oai_quant(
    x, alpha, limit, need_quant=True, group_list=None, group_list_type=None,
    *, impl="auto",
):
    ...
```

`auto` selects `triton`. If the selected implementation is unavailable, the call raises an error; it does not fall back to another backend.

## Description

Applies clamped SwiGLU-OAI to a split `[gate | up]` input, optionally followed by per-row symmetric INT8 quantization. The two projections occupy the first and second halves of the last dimension; they are not interleaved.

Let the last dimension be $2D$, and flatten the leading dimensions into $S$ rows. For each processed row:

$$
g_j=\min(x_j,\mathrm{limit}),\qquad
u_j=\max(-\mathrm{limit},\min(x_{D+j},\mathrm{limit})),
$$

$$
y_j=g_j\,\sigma(g_j\,\mathrm{alpha})\,(u_j+1).
$$

The gate has only an upper clamp. Activation arithmetic uses float32. Without quantization, the result is cast to `x.dtype`. With quantization, each row has its own scale. For a nonzero scale:

$$
s=\frac{\max_j|y_j|}{127},\qquad
q_j=\mathrm{sat}_{[-128,127]}\left(\mathrm{trunc}\left(
\mathrm{cast}_{\mathrm{x.dtype}}(y_j/s)\right)\right).
$$

Normalized values are converted to `x.dtype` before truncation toward zero and saturation to INT8. For an all-zero activation row, both the quantized row and its scale are zero. Recover approximate floating-point values with `q.float() * s`; the scale is not its reciprocal. Inputs and activation intermediates must be finite. This operator is intended for inference; gradient computation is not guaranteed.

Without `group_list`, all $S$ rows are processed. With `group_list`, only the first $R$ flattened rows are processed:

- `group_list_type=0`: cumulative token counts without an initial sentinel; `R = group_list[-1]`.
- `group_list_type=1`: token counts per expert; `R = group_list.sum()`.

Zero-token experts are allowed. The operator does not reorder tokens. Counts must be nonnegative, and cumulative counts must also be nondecreasing. Callers must ensure `0 <= R <= S` and that counts and totals fit signed int32. Triton does not copy group values to the host to validate them.

## Parameters and return values

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[..., 2D]` | `torch.float32`, `torch.float16`, `torch.bfloat16` | Required | Strided tensor with rank at least 1 and `D > 0`. Leading dimensions may be empty. Non-contiguous views are supported through reshape/contiguous copies. |
| `alpha` | `int` or `float` | — | — | Required | Finite sigmoid multiplier, representable in float32. |
| `limit` | `int` or `float` | — | — | Required | Finite positive clamp bound, representable in float32. |
| `need_quant` | `bool` | — | — | `True` | Enable INT8 output and per-row scales. Triton requires `D` to be a multiple of 1536 when True. |
| `group_list` | `torch.Tensor` or `None` | `[E]` | `torch.int32`, `torch.int64` | `None` | Optional nonempty 1-D strided tensor on x.device. Counts or cumulative counts for the processed prefix. |
| `group_list_type` | `int` or `None` | — | — | `None` | Required with group_list: `0` for cumulative counts, `1` for counts. Ignored in dense mode. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"triton"`, or `"torch_native"`. |

An invalid group encoding or index dtype raises `ValueError`. Callers are responsible for the other input constraints.

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `out` | `torch.Tensor` | `[..., D]` | `torch.int8` if need_quant, otherwise x.dtype | New output on x.device. Only the first R flattened rows are defined in grouped mode. |
| `scale` | `torch.Tensor` | `[S]` | `torch.float32` | New scale buffer on x.device. Only the first R entries are defined when quantization is enabled. All entries are unspecified when need_quant=False. |

Returns `(out, scale)` without modifying inputs. In dense mode, use `R=S`. Unprocessed rows must not be consumed or included in accuracy comparisons. Empty inputs return empty buffers without launching a kernel.

## triton backend

- Runs on an Ascend NPU. Select `x.device` as the current NPU before calling the operator.
- With quantization enabled, output width `D` must be a multiple of 1536. This restriction does not apply when quantization is disabled.
- Each activation row is processed together and must fit in on-chip memory. A width divisible by 1536 does not imply support for arbitrarily wide rows.
- Group counts are read on the device. Non-contiguous inputs may require a copy.

## torch_native backend

- Uses PyTorch operations as a reference implementation and supports any positive output width `D`.
- Reading the valid row count in grouped mode may synchronize the device.
- Floating-point differences between backends can move values across an INT8 truncation boundary.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import swiglu_oai_quant

x = torch.randn(17, 3072, device="npu", dtype=torch.float32)
out, scale = swiglu_oai_quant(x, 1.702, 7.0)
reference, scale_ref = swiglu_oai_quant(x, 1.702, 7.0, impl="torch_native")
torch.testing.assert_close(out, reference, rtol=0, atol=1)
torch.testing.assert_close(scale, scale_ref, rtol=5e-3, atol=0)

groups = torch.tensor([0, 2, 0, 3], device="npu", dtype=torch.int64)
grouped_out, _ = swiglu_oai_quant(
    x, 1.702, 7.0, need_quant=False, group_list=groups, group_list_type=1
)
valid_output = grouped_out[:5]  # Remaining rows and all scales are unspecified.
```

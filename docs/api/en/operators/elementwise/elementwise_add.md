# elementwise_add

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Supported | Unverified |
| `torch_native` | Supported | Unverified |

## Signature

The default option is `auto`, registered to the same entry point as `ascendc`. Omitting `impl` or passing `"auto"` selects the Ascend C operator.

```python
from sgl_kernel_npu.operators.elementwise import elementwise_add

def elementwise_add(x, y, *, impl="auto"):
    ...
```

## Description

Adds two tensors element-wise and returns a newly allocated tensor:

$$
Y_{i_0,\dots,i_{n-1}} = X_{i_0,\dots,i_{n-1}} + Z_{i_0,\dots,i_{n-1}}
$$

The Ascend C implementation flattens the inputs to a one-dimensional view, so
the rank of the inputs does not change the kernel path. The element count is
aligned up to `blockDim * tileLength` on the host by zero padding; the padding is
trimmed from the result, which is then reshaped back to the input shape. This
keeps the kernel free of tail handling and keeps `DataCopy` 32 B aligned.

fp16 inputs accumulate in fp16 (the vector `Add` runs in the input dtype).
Inputs that may overflow fp16 should be added through `impl="torch_native"`
(or an fp32 tensor), which follows the usual torch promotion rules.

The operator is intended as a sample: see
[`docs/developer_guide/ascendc_operator_sample.md`](../../../../developer_guide/ascendc_operator_sample.md)
for the full development and verification workflow.

## Parameters and return value

The public operator forwards arguments without broadcasting or implicit
conversion; the Ascend C launcher validates them.

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | any rank | `torch.float16`, `torch.float32` | Required | First addend. Must be contiguous and on the NPU. |
| `y` | `torch.Tensor` | same as `x` | same as `x` | Required | Second addend. Must be contiguous, on the same device, with the same shape and dtype as `x`. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"ascendc"`, or `"torch_native"`. `auto` currently selects the Ascend C implementation. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | same as `x` | same as `x` | Newly allocated; inputs are not modified. |

## Backend descriptions

### ascendc

Vector-unit (AI Core) implementation in `csrc/elementwise_add`. It supports
contiguous fp16/fp32 tensors of any rank and size, including sizes that are not
multiples of the alignment unit. Empty inputs return an empty tensor without
launching a kernel. Non-contiguous, mismatched-shape, mismatched-dtype, and
bfloat16 inputs raise `RuntimeError` before any device work is queued.

### torch_native

`x + y` through regular torch dispatch. It serves as the accuracy reference and
supports whatever dtype and broadcasting the installed torch build supports, so
it is not restricted to the Ascend C input contract.

## Example

```python
import torch
import torch_npu  # noqa: F401  (registers the NPU device)
from sgl_kernel_npu.operators.elementwise import elementwise_add

x = torch.randn((1, 4096), dtype=torch.float16, device="npu")
y = torch.randn((1, 4096), dtype=torch.float16, device="npu")

out = elementwise_add(x, y)                      # auto -> ascendc
ref = elementwise_add(x, y, impl="torch_native")  # reference path
torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
```

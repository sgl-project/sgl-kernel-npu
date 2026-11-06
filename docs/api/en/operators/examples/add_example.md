# add_example

## Hardware support

| Implementation | arch22 (2201, A2/A3) | arch35 (3510, A5) |
| --- | --- | --- |
| `aicpu` | Supported | Supported |
| `torch_native` | Supported | Supported |

`auto` is an alias of `aicpu`, not a separate implementation. Install a wheel
built for the device architecture and including `add_example` in its `--ops`
selection. See the [build instructions](../../../../../python/sgl_kernel_npu/README.md#building-and-installing-aicpu-operators).

## Signature

```python
from sgl_kernel_npu.operators.examples import add_example

def add_example(x1, x2, *, impl="auto"):
    ...
```

Omitting `impl` or passing `"auto"` selects `aicpu`. No automatic fallback occurs
when the native operator is unavailable. Pass `impl="torch_native"` explicitly
to select the independent reference.

## Description

Returns a new tensor containing the elementwise sum of two inputs:

$$
y_i = (x_1)_i + (x_2)_i.
$$

The inputs have identical shapes and dtypes; broadcasting is rejected. Both
inputs remain unchanged. The result has independent storage, the same shape,
dtype and NPU device as the inputs, and contiguous ND storage.

This is an AICPU integration example, not a performance benchmark. The example
uses values without integer overflow; overflow behavior is not validated.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `x1` | `torch.Tensor` | `S` | `torch.float32` or `torch.int32` | Required | NPU tensor with at least one element; contiguous ND storage. |
| `x2` | `torch.Tensor` | Same `S` as `x1` | Same as `x1` | Required | On the same NPU as `x1`, with contiguous ND storage. Broadcasting is not supported. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"aicpu"` or `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `y` | `torch.Tensor` | Same `S` as the inputs | Same as the inputs | Newly allocated, contiguous ND output on the same NPU; does not alias either input. |

The public interface checks shape, dtype, device, nonempty input and storage
layout before dispatch. Unsupported inputs are rejected rather than converted.

## aicpu backend

Calls the packaged native `torch.ops.npu.sgl_add_example` binding. Supports
matching FP32 or INT32 inputs under the constraints above. The wheel must include
this native operator; a missing binding causes an error.

## torch_native backend

Uses `torch.add` as an independent reference. The public interface applies the
same input contract, including the same-NPU requirement.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.examples import add_example

x1 = torch.tensor([2, 5], dtype=torch.float32, device="npu")
x2 = torch.tensor([3, 3], dtype=torch.float32, device=x1.device)
result = add_example(x1, x2)  # auto -> aicpu; result is [5, 8]
reference = add_example(x1, x2, impl="torch_native")
torch.testing.assert_close(result, reference, atol=0, rtol=0)
```

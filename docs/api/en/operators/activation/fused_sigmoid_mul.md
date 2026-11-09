# fused_sigmoid_mul

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

Omitting `impl` or passing `"auto"` selects `"triton"` without automatic fallback.

```python
from sgl_kernel_npu.operators.activation import fused_sigmoid_mul

def fused_sigmoid_mul(x, gate, *, impl="auto"):
    ...
```

## Description

Computes `x * sigmoid(gate)` element by element. For example, a gate value of 0
halves the corresponding value in `x`. The sigmoid and multiplication run in
FP32, then the result is cast back to `x.dtype`.

SGLang uses this operator in the full-attention layers of both dense and MoE
Qwen3.5 models, before the output projection. When output gating is enabled,
the standard NPU path passes the attention output and `gate` in matching shapes.
[See the SGLang call site](https://github.com/sgl-project/sglang/blob/53a24c448ecd6c26bf6461b80ffb8d55397d57cf/python/sglang/srt/models/qwen3_5.py#L1576-L1587)

Intern-S2-Preview, Intern-S2-Mobius, and MiniCPM-V 4.6 also reuse this attention path.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | Any shape, including scalars and empty tensors | `torch.float32`, `torch.float16`, `torch.bfloat16` | Required | NPU input with `torch.strided` layout; contiguous and non-contiguous views are accepted. |
| `gate` | `torch.Tensor` | Same as `x` | `torch.float32`, `torch.float16`, `torch.bfloat16` | Required | Sigmoid input on the same NPU as `x`, with `torch.strided` layout. Its dtype may differ from `x`; non-contiguous views are accepted. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"triton"`, or `"torch_native"`. `auto` selects Triton. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | Same as `x` | `x.dtype` | Newly allocated on `x.device`; inputs are not modified. Empty inputs produce an empty tensor. Triton returns contiguous output; torch_native follows PyTorch's output layout. |

Mismatched input shapes raise `ValueError`. Dtype, layout, and device requirements
are caller preconditions and are not explicitly validated on each call.

## triton backend

- Makes both inputs contiguous before applying elementwise sigmoid gating.
- Computes in FP32 and casts the output to `x.dtype`, including mixed-dtype inputs.
- Provides forward execution for inference. Set the current NPU to the device
  holding the inputs.

## torch_native backend

- Uses independent PyTorch operations, converting each input to FP32 and casting
  the result to `x.dtype`.
- Supports non-contiguous inputs; the output layout follows PyTorch operations.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import fused_sigmoid_mul

x = torch.tensor([2.0, -4.0], device="npu", dtype=torch.bfloat16)
gate = torch.zeros_like(x)
y = fused_sigmoid_mul(x, gate)  # auto selects Triton; [1.0, -2.0]
reference = fused_sigmoid_mul(x, gate, impl="torch_native")
torch.testing.assert_close(y, reference)
```

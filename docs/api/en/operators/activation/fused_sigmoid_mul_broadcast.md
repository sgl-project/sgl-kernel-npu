# fused_sigmoid_mul_broadcast

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

Omitting `impl` or passing `"auto"` selects `"triton"`.

```python
from sgl_kernel_npu.operators.activation import fused_sigmoid_mul_broadcast

def fused_sigmoid_mul_broadcast(x, gate, *, impl="auto"):
    ...
```

## Description

Applies one gate per row. For input $X$ of shape $[N, D]$ and gate $g$ of
shape $[N]$ or $[N, 1]$, with $g_n$ denoting the gate value for row $n$:

$$
Y_{n,j} = X_{n,j} \cdot \sigma(g_n), \qquad
\sigma(z) = \frac{1}{1 + e^{-z}}.
$$

A gate value of zero halves the row. Both backends convert the inputs to
FP32 for computation and cast the result to `x.dtype`.

In SGLang, the NPU branch of `Qwen2MoeSparseMoeBlock._forward_shared_experts`
in `sglang/srt/models/qwen2_moe.py` calls this operator when the shared expert
runs separately and needs gating. Qwen2-MoE, Qwen3-Next, Qwen3.5-MoE,
Qwen4Exp, and InternS2Preview can reach this branch through the same block.
The model passes a shared-expert output of shape `[tokens, hidden_size]` and a
gate of shape `[tokens, 1]`. SGLang currently imports the legacy entry point.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[N, D]` | `torch.float32`, `torch.float16`, `torch.bfloat16` | Required | NPU input; use `N > 0` and `D > 0`. Contiguous and non-contiguous strided tensors are accepted. |
| `gate` | `torch.Tensor` | `[N]` or `[N, 1]` | Same as `x` | Required | One gate logit per row, on the same NPU as `x`. Contiguous and non-contiguous strided tensors are accepted. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"triton"`, or `"torch_native"`. `auto` selects Triton. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[N, D]` | `x.dtype` | Newly allocated on `x.device`; inputs are not modified. Triton returns a contiguous tensor; torch_native follows PyTorch's output layout. |

Incompatible input ranks or gate shapes raise `ValueError`. Dtype, device,
layout, and positive-dimension requirements are caller preconditions and are
not explicitly validated on each call. Other dtypes are outside the documented
input contract.

## triton backend

- Computes row-wise sigmoid gating with FP32 intermediates.
- Accepts non-contiguous inputs by making them contiguous before computation;
  the output is contiguous.

## torch_native backend

- Uses independent PyTorch operations with FP32 intermediates.
- Accepts non-contiguous inputs; the output layout follows PyTorch operations.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import fused_sigmoid_mul_broadcast

x = torch.tensor([[2.0, 4.0], [6.0, 8.0]], device="npu")
gate = torch.zeros(2, device=x.device, dtype=x.dtype)
output = fused_sigmoid_mul_broadcast(x, gate)  # auto selects Triton
reference = fused_sigmoid_mul_broadcast(x, gate, impl="torch_native")
torch.testing.assert_close(output, x / 2)
torch.testing.assert_close(output, reference)
```

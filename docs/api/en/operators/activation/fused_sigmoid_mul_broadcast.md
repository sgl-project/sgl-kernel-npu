# fused_sigmoid_mul_broadcast

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Verified | Verified |
| `torch_native` | Verified | Verified |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

## Signature

```python
from sgl_kernel_npu.operators.activation import fused_sigmoid_mul_broadcast

def fused_sigmoid_mul_broadcast(x, gate, *, impl="auto"):
    ...
```

## Description

Applies one gate per row of `x`. The operator computes the sigmoid of each
gate value, then multiplies every element in that row by the result. A gate
value of zero halves the row. Both backends compute in FP32 and cast the
result to `x.dtype`.

In SGLang, the NPU branch of `Qwen2MoeSparseMoeBlock._forward_shared_experts`
in `sglang/srt/models/qwen2_moe.py` calls this operator when the shared expert
runs separately and needs gating. Qwen2-MoE, Qwen3-Next, Qwen3.5-MoE,
Qwen4Exp, and InternS2Preview can reach this branch through the same block.
The model passes a shared-expert output of shape `[tokens, hidden_size]` and a
gate of shape `[tokens, 1]`. SGLang currently imports the legacy entry point.

## Parameters and return value

| Parameter | Shape | Description |
| --- | --- | --- |
| `x` | `[N, D]` | NPU tensor of dtype `torch.float32`, `torch.float16`, or `torch.bfloat16`. Use `N > 0` and `D > 0`. |
| `gate` | `[N]` or `[N, 1]` | One value per row, on the same NPU as `x`. |
| `impl` | — | Keyword-only. `"auto"` (default) selects `"triton"`. Use `"torch_native"` for the PyTorch implementation. |

Returns a new `[N, D]` tensor with the same dtype and device as `x`. Both
backends accept non-contiguous inputs. Triton returns a contiguous tensor.
The PyTorch implementation follows PyTorch's output layout.

The public `_check_inputs` retains only the legacy input-rank and gate-shape
checks, raising `ValueError` for incompatible shapes. Both backends share these
checks without repeating them internally. Dtype and device requirements above
are caller preconditions and are not explicitly validated on each call.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import fused_sigmoid_mul_broadcast

x = torch.tensor([[2.0, 4.0], [6.0, 8.0]], device="npu")
gate = torch.zeros(2, device=x.device)
output = fused_sigmoid_mul_broadcast(x, gate)
torch.testing.assert_close(output, x / 2)
```

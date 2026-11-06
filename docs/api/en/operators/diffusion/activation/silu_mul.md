# silu_mul

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Supported |

## Function prototype

The default is `auto`, which is registered to the same entry as `triton`. Omitting `impl` or passing `"auto"` uses Triton.

```python
from sgl_kernel_npu.operators.diffusion import silu_mul

def silu_mul(a, b, *, impl="auto"):
    ...
```

## Behavior

Computes `silu(a) * b` elementwise and reproduces the two BF16 roundings of the eager chain:

$$
s = \operatorname{round}_{bf16}\left(a \cdot \sigma(a)\right),\qquad
y = \operatorname{round}_{bf16}\left(s \cdot b\right)
$$

where $\sigma$ is the sigmoid. The result is bitwise identical to `torch.nn.functional.silu(a) * b`. The Triton backend performs both roundings in one kernel without writing intermediates back to global memory. The operator originated in Qwen-Image 2.1's SwiGLU FFN but applies to any BF16 silu-gated multiply that must match eager bit for bit.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `a` | `torch.Tensor` | `[*S]` | `torch.bfloat16` | required | Gate input; contiguous tensor on an NPU. |
| `b` | `torch.Tensor` | `[*S]` | `torch.bfloat16` | required | Same shape, device and layout as `a`. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; one of `"auto"`, `"triton"`, `"torch_native"`. |

| Return | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[*S]` | `torch.bfloat16` | Newly allocated tensor on `a`'s device; inputs are not modified. |

## triton backend

- Each program walks several contiguous 4096-element tiles; the tail is a separate masked kernel.
- Requires same-shape, same-device, contiguous BF16 inputs with `0 < numel < 2**31`; otherwise raises `ValueError` before launch.
- The two BF16 roundings are an explicit `vcast` and the final store.

## torch_native backend

- Evaluates `torch.nn.functional.silu(a) * b` directly and serves as the independent accuracy reference.

## Example

```python
import torch

from sgl_kernel_npu.operators.diffusion import silu_mul

a = torch.randn((1, 4096, 12288), dtype=torch.bfloat16, device="npu")
b = torch.randn_like(a)
out = silu_mul(a, b)
ref = torch.nn.functional.silu(a) * b
assert torch.equal(out, ref)
```

# residual_gate_add

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Supported |

## Function prototype

The default is `auto`, which is registered to the same entry as `triton`. Omitting `impl` or passing `"auto"` uses Triton.

```python
from sgl_kernel_npu.operators.diffusion import residual_gate_add

def residual_gate_add(residual, update, gate, *, impl="auto"):
    ...
```

## Behavior

Computes `residual + update * gate` and reproduces the two BF16 roundings of the eager chain:

$$
p = \operatorname{round}_{bf16}\left(update \odot gate\right),\qquad
y = \operatorname{round}_{bf16}\left(residual + p\right)
$$

The result is bitwise identical to separate eager BF16 multiply and add operations. `gate` is either an elementwise gate with `residual`'s shape or a single broadcast row (`[1, C]` or `[C]`), which is the per-timestep modulation used by diffusion models. The operator originated in Qwen-Image 2.1's DiT block residual update.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `residual` | `torch.Tensor` | `[..., C]` | `torch.bfloat16` | required | Accumulator; contiguous tensor on an NPU, at least 2-D. |
| `update` | `torch.Tensor` | `[..., C]` | `torch.bfloat16` | required | Same shape, device and layout as `residual`. |
| `gate` | `torch.Tensor` | `[..., C]` or `[C]` | `torch.bfloat16` | required | Elementwise gate matching `residual`, or one broadcast row. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; one of `"auto"`, `"triton"`, `"torch_native"`. |

| Return | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[..., C]` | `torch.bfloat16` | Newly allocated tensor on `residual`'s device; inputs are not modified. |

## triton backend

- Processes `[ROWS, C]` tiles; `ROWS` is clamped by the UB budget and by row divisibility, and each program walks several tiles.
- Requires a power-of-two last dimension and at least 16 rows per tile with an exact row split; otherwise raises `ValueError` before launch.
- Accumulates in FP32, rounds the product to BF16 explicitly, then adds in FP32 and rounds once on store.

## torch_native backend

- Evaluates `residual + update * gate` directly and serves as the independent accuracy reference.

## Example

```python
import torch

from sgl_kernel_npu.operators.diffusion import residual_gate_add

residual = torch.randn((1, 4096, 4096), dtype=torch.bfloat16, device="npu")
update = torch.randn_like(residual)
gate = torch.randn((4096,), dtype=torch.bfloat16, device="npu")
out = residual_gate_add(residual, update, gate)
ref = residual + update * gate
assert torch.equal(out, ref)
```

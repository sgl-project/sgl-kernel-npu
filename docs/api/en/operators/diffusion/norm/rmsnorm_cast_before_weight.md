# rmsnorm_cast_before_weight

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Supported |

## Function prototype

The default is `auto`, which is registered to the same entry as `triton`. Omitting `impl` or passing `"auto"` uses Triton.

```python
from sgl_kernel_npu.operators.diffusion import rmsnorm_cast_before_weight

def rmsnorm_cast_before_weight(x, weight, eps, *, impl="auto"):
    ...
```

## Behavior

Applies RMSNorm over the last dimension and reproduces the rounding order of `RMSNorm(cast_x_before_out_mul=True)`:

$$
v_i = \frac{1}{D}\sum_{j=0}^{D-1} x_{ij}^2
$$

$$
n_{ij} = \operatorname{round}_{dtype}\left(x_{ij} \cdot \frac{1}{\sqrt{v_i+\varepsilon}}\right),\qquad
y_{ij} = \operatorname{round}_{dtype}\left(n_{ij} \cdot w_j\right)
$$

The normalized value is converted back to the input dtype before the weight multiply, which differs from interfaces such as `rmsnorm_bias` that multiply in FP32. The operator originated in Qwen-Image 2.1's Q/K normalization.

The Triton backend reduces rows with `tl.sum`, which matches aten `mean`'s reduction tree on the validated NPU architecture and therefore reproduces it bit for bit. Other architectures require revalidation (see the hardware table).

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[..., D]` | `torch.bfloat16`, `torch.float16` | required | Input; contiguous on an NPU, with `D` a power of two. |
| `weight` | `torch.Tensor` | `[D]` | same as `x` | required | Per-channel weight; same device as `x`, contiguous. |
| `eps` | `float` | — | — | required | Stability term added to the mean square. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; one of `"auto"`, `"triton"`, `"torch_native"`. |

| Return | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[..., D]` | same as `x` | Newly allocated tensor on `x`'s device; inputs are not modified. |

## triton backend

- Each program handles `[ROWS, D]` tiles; `ROWS` is clamped by the UB budget and row divisibility, and the reduction stays inside the tile.
- Requires a power-of-two `D` and at least 16 rows per tile with an exact row split; otherwise raises `ValueError` before launch.
- Reduces with `tl.sum`, casts the normalized value explicitly before multiplying the weight, and rounds once on store.

## torch_native backend

- Evaluates `x.float() → square().mean(-1) → rsqrt → cast → weight multiply` step by step and serves as the independent accuracy reference.

## Example

```python
import torch

from sgl_kernel_npu.operators.diffusion import rmsnorm_cast_before_weight

x = torch.randn((1, 4096, 32, 128), dtype=torch.bfloat16, device="npu")
weight = torch.randn((128,), dtype=torch.bfloat16, device="npu")
out = rmsnorm_cast_before_weight(x, weight, 1e-6)

values = x.float()
variance = values.square().mean(dim=-1, keepdim=True)
ref = weight * (values * torch.rsqrt(variance + 1e-6)).to(x.dtype)
assert torch.equal(out, ref)
```

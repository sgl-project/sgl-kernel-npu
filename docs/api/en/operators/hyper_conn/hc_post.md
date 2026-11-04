# hc_post

Upstream documentation: [recipes npu_hc_post](https://gitcode.com/cann/cann-recipes-infer/blob/35a476c2b1ea61990380a7115ea991023aedc7d0/ops/ascendc/docs/custom-npu_hc_post.md).

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `cann_recipes_infer_cann910` / `auto` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

The default `auto` selects `cann_recipes_infer_cann910` on CANN 9.1.0 and raises on other CANN releases.

```python
from sgl_kernel_npu.operators.hyper_conn import hc_post

def hc_post(x, residual, post, comb, *, impl="auto"):
    ...
```

## Description

Mixes the residual streams and adds the weighted layer output. For each output stream `j` and hidden channel `d`:

$$
Y_{\ldots,j,d}
= P_{\ldots,j}X_{\ldots,d}
+ \sum_{i=0}^{3} C_{\ldots,i,j}R_{\ldots,i,d}
$$

`X`, `R`, `P`, and `C` correspond to `x`, `residual`, `post`, and `comb`. The operator sums over the first stream axis of `comb`; pass the mixing matrix without transposing it. Inputs are not modified.

## Parameters and return value

All inputs must be contiguous and on the same NPU. `T` is the token count; alternatively, tokens can retain batch and sequence dimensions `[B,S]`. All dimensions must be positive. There are four residual streams, and `D` must be 4096 or 7168.

| Parameter | Type / shape / dtype | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `x` | Tensor `[T,D]` or `[B,S,D]`; float16, bfloat16, or float32 | Required | Current layer output. |
| `residual` | Tensor `[T,4,D]` or `[B,S,4,D]`; same dtype as `x` | Required | Residual streams; leading dimensions and `D` must match `x`. |
| `post` | Tensor `[T,4]` or `[B,S,4]`; float16, bfloat16, or float32 | Required | Weight applied to `x` for each output stream. Leading dimensions must match `x`. |
| `comb` | Tensor `[T,4,4]` or `[B,S,4,4]`; same dtype as `post` | Required | Residual mixing weights, indexed by input stream then output stream. Leading dimensions must match `x`. |
| `impl` | `str`, keyword-only | `"auto"` | `auto`, `cann_recipes_infer_cann910`, or `torch_native`. |

| Return value | Type / shape / dtype | Meaning |
| --- | --- | --- |
| `output` | Tensor with the shape, dtype, and device of `residual` | Newly allocated mixed residual streams. |

## cann_recipes_infer_cann910 backend

Wraps the recipes `npu_hc_post` interface. Use CANN 9.1.0 and an sgl-kernel-npu wheel built with the `cann_recipes_infer` profile. The wheel includes the selected kernel and precompiled Torch binding; no separate recipes installation is required.

## torch_native backend

Uses PyTorch operations with float32 intermediates and casts the output to `x.dtype`. It follows the same shape and dtype constraints. Select it explicitly on other CANN releases; `auto` does not fall back to it. Floating-point accumulation may differ from the recipes implementation.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.hyper_conn import hc_post

x = torch.randn(1, 4096, device="npu", dtype=torch.bfloat16)
residual = torch.randn(1, 4, 4096, device="npu", dtype=torch.bfloat16)
post = torch.randn(1, 4, device="npu", dtype=torch.float32)
comb = torch.randn(1, 4, 4, device="npu", dtype=torch.float32)

output = hc_post(x, residual, post, comb)  # CANN 9.1.0, recipes-enabled wheel
output_native = hc_post(x, residual, post, comb, impl="torch_native")
print(output.shape)  # torch.Size([1, 4, 4096])
```

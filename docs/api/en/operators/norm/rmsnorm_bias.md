# rmsnorm_bias

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Unverified |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

## Signature

The default option is `auto`, currently registered to the same entry point as `triton`. Omitting `impl` or passing `"auto"` selects Triton.

```python
from sgl_kernel_npu.operators.norm import rmsnorm_bias

def rmsnorm_bias(
    input,
    norm_weight,
    norm_bias,
    eps,
    quant_scale=None,
    quant_offset=None,
    *,
    impl="auto",
):
    ...
```

## Description

Applies RMS normalization to each row of a two-dimensional input, then multiplies by the weight and adds the bias. For input $X \in \mathbb{R}^{B\times H}$, weight $w$, and bias $b$:

$$
v_i = \frac{1}{H}\sum_{j=0}^{H-1}X_{ij}^{2}
$$

$$
Y_{ij} = \frac{X_{ij}}{\sqrt{v_i+\varepsilon}}w_j+b_j
$$

Without quantization, the result is cast to the input dtype. Providing `quant_scale` enables int8 quantization:

$$
Q_{ij} = \operatorname{cast}^{\mathrm{saturate}}_{\mathrm{int8}}
\left(Y_{ij}s_j+o_j\right)
$$

Here, $s$ is `quant_scale` and $o$ is `quant_offset`. The torch_native implementation truncates toward zero and saturates to `[-128, 127]`, following the [Triton Ascend conversion tests](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/pytest_ut/test_cast_full.py).

## Parameters and return value

The tables describe the recommended calling convention and current implementation requirements. The public operator forwards arguments without input validation.

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `input` | `torch.Tensor` | `[B, H]` | `torch.float32`, `torch.float16`, `torch.bfloat16` | Required | Input to normalize; use `B > 0, H > 0`. Triton requires a contiguous two-dimensional NPU tensor. |
| `norm_weight` | `torch.Tensor` | `[H]` | Floating point; preferably the same as input | Required | Per-channel multiplicative weight, on the same device as input. Triton reads it as a contiguous vector. |
| `norm_bias` | `torch.Tensor` | `[H]` | Floating point; preferably the same as input | Required | Per-channel additive bias, on the same device as input. Triton reads it as a contiguous vector. |
| `eps` | `float` | — | — | Required | Stability term added to the mean square; a finite positive value such as `1e-6` is recommended. |
| `quant_scale` | `torch.Tensor` or `None` | `[H]` or — | Floating point; `torch.float32` recommended | `None` | Enables quantization when not None; quant_offset must also be provided. Must be on the same device as input; Triton requires contiguous storage. |
| `quant_offset` | `torch.Tensor` or `None` | `[H]` or — | Floating point; `torch.float32` recommended | `None` | Additive quantization offset, on the same device as input; Triton requires contiguous storage. Ignored when quant_scale is None. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"triton"`, or `"torch_native"`. `auto` currently selects the Triton implementation. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[B, H]` | input.dtype without quantization; `torch.int8` with quantization | Newly allocated output on the same device as input. Inputs are not modified. |

Use finite inputs and quantization intermediates; cross-backend NaN conversion behavior is unspecified. The public interface neither validates inputs nor switches implementations automatically. Package initialization still requires `torch_npu` and the native shared library.

## triton backend

- Uses a CANN Triton kernel for non-quantized and int8-quantized output; requires contiguous inputs.
- Quantization uses fixed 1024-element slices. The full supported hidden-size range remains unverified.

## torch_native backend

- Uses PyTorch operations for non-quantized and int8-quantized output, keeping intermediates in float32 until quantization; supports non-contiguous inputs.
- Follows the same computation flow and quantization branch condition. Reduction order and fused multiply-add operations may cause numerical differences; bitwise equality near quantization boundaries is not guaranteed.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.norm import rmsnorm_bias

x = torch.randn(4, 1024, device="npu", dtype=torch.float32)
weight = torch.randn(1024, device="npu", dtype=torch.float32)
bias = torch.randn(1024, device="npu", dtype=torch.float32)

print(list_impls(rmsnorm_bias))  # ['auto', 'torch_native', 'triton']
y_default = rmsnorm_bias(x, weight, bias, 1e-6)  # Default auto selects Triton
y_auto = rmsnorm_bias(x, weight, bias, 1e-6, impl="auto")
y_triton = rmsnorm_bias(x, weight, bias, 1e-6, impl="triton")
y_native = rmsnorm_bias(x, weight, bias, 1e-6, impl="torch_native")
torch.testing.assert_close(y_default, y_triton, rtol=0, atol=0)
torch.testing.assert_close(y_auto, y_triton, rtol=0, atol=0)
torch.testing.assert_close(y_triton, y_native, rtol=1e-4, atol=1e-4)
```

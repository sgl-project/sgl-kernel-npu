# swiglu_group_quant

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Not supported | Supported |
| `torch_native` | Supported | Supported |
| `triton` | Not supported | Not supported |
| `tilelang` | Not supported | Not supported |

## Signature

```python
from sgl_kernel_npu.operators.activation import swiglu_group_quant

def swiglu_group_quant(x, topk_weight=None, group_index=None, dst_type=None, quant_mode=1, group_size=128, round_scale=False, ue8m0_scale=False, output_origin=False, group_list_type=0, clamp_value=0.0, *, impl='auto'): ...
```

`auto` selects `ascendc`.

## Description

Apply SwiGLU and quantize expert activations.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `x` | Tensor | [...,2C] | FP16/BF16 | Required | Contiguous FP16/BF16 gate and up projections concatenated on the last axis. Width must be divisible by 256 for modes 1/3 or 128 for MX. |
| `topk_weight` | Tensor | [N] | FP32 | None | Optional float32 routing weight per row. |
| `group_index` | Tensor | [E] | int64 | None | Optional int64 counts per expert. Only the leading sum of counts is processed. |
| `dst_type` | torch.dtype | — | — | None | FP8 E4M3FN (default) or E5M2. |
| `quant_mode` | int | — | — | 1 | 1 for group quantization, 2 for MX, 3 for per-token FP8 blocks. |
| `group_size` | int | — | — | 128 | Accepted for compatibility. Modes 1/3 use 128 values per scale, MX uses 32. |
| `round_scale` | bool | — | — | False | Use power-of-two scales in mode 3. |
| `ue8m0_scale` | bool | — | — | False | Store mode 3 rounded scales in E8M0. Use with round_scale. |
| `output_origin` | bool | — | — | False | Populate the unquantized output in mode 3. |
| `group_list_type` | int | — | — | 0 | 0 for per-expert counts. |
| `clamp_value` | float | — | — | 0.0 | Nonzero limit for gate (upper bound) and up (both bounds). |
| `impl` | str | — | — | 'auto' | auto or ascendc on A5, torch_native, or cpu. |

| Return value | Type | Description |
| --- | --- | --- |
| `y` | Tensor | FP8 with shape [..., C]. |
| `scale` | Tensor | MX returns E8M0 [..., C/64, 2]. Modes 1/3 return [..., C/128]. |
| `y_origin` | Tensor | Input dtype and shape [..., C]. Defined only in mode 3 with output_origin=True. |

## ascendc backend

Runs fused activation and quantization on A5.

## torch_native backend

Uses PyTorch operations on the input devices.

## cpu backend

Computes and returns tensors on CPU.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.activation import swiglu_group_quant

x = torch.randn(4, 512, device="npu", dtype=torch.bfloat16)
counts = torch.tensor([1, 0, 3], device="npu")
y, scale, _ = swiglu_group_quant(x, group_index=counts, quant_mode=2)
print(y.shape, scale.shape)
```

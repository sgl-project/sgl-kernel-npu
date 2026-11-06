# swiglu_group_quant

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Unsupported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.activation import swiglu_group_quant

def swiglu_group_quant(x, topk_weight=None, group_index=None, dst_type=None, quant_mode=1, group_size=128, round_scale=False, ue8m0_scale=False, output_origin=False, group_list_type=0, clamp_value=0.0, *, impl='auto'): ...
```

`auto` selects `ascendc` on arch35 and does not fall back on unsupported hardware.

## Description

Apply SwiGLU activation and quantize its output for an expert FFN, including the DeepSeek-V4 MXFP8 path. The last input dimension contains gate and up projections; the operation computes SiLU(gate) × up, with optional input clamping and per-row routing weights.

The quantization mode selects the scale rule:

- Mode 1: FP8 group quantization with one FP32 scale per 128 activation values, based on the target FP8 format.
- Mode 2: MXFP8 quantization with one E8M0 scale per 32 values, derived from activation exponents after rounding to the input dtype.
- Mode 3: FP8 block quantization with one scale per 128 values, a maximum-value floor of 1e-4 and a fixed scale reference of 448. round_scale rounds the scale up to a power of two; ue8m0_scale can store these rounded scales as E8M0.

The function returns quantized y, its scale tensor and y_origin. The latter is populated only in mode 3 with output_origin=True. group_index limits the processed prefix using per-expert token counts.

## Parameters and return value

N is the product of the leading dimensions of x. For NPU implementations, put all tensors on the same NPU and use contiguous inputs. None or empty optional tensors are treated as absent. Inputs are not modified.

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `x` | Tensor | [...,2C] | FP16/BF16 | Required | Contiguous FP16/BF16 gate and up projections concatenated on the last axis. Width must be divisible by 256 for modes 1/3 or 128 for MX. |
| `topk_weight` | Tensor | [N] | FP32 | None | Optional float32 routing weight per row. |
| `group_index` | Tensor | [E] | int64 | None | Optional int64 counts per expert. Counts must be nonnegative. Process min(N, sum(counts)) leading rows; remaining output rows have unspecified values. |
| `dst_type` | torch.dtype | — | — | None | FP8 E4M3FN (default) or E5M2. |
| `quant_mode` | int | — | — | 1 | 1 for group quantization, 2 for MX, 3 for per-token FP8 blocks. |
| `group_size` | int | — | — | 128 | Accepted for compatibility. Modes 1/3 use 128 values per scale, MX uses 32. |
| `round_scale` | bool | — | — | False | Use power-of-two scales in mode 3. |
| `ue8m0_scale` | bool | — | — | False | Store mode 3 rounded scales in E8M0. Use with round_scale. |
| `output_origin` | bool | — | — | False | Populate the unquantized output in mode 3. |
| `group_list_type` | int | — | — | 0 | 0 for per-expert counts. |
| `clamp_value` | float | — | — | 0.0 | 0 disables clamping. A positive value bounds gate above and up on both sides before activation. |
| `impl` | str | — | — | 'auto' | auto, ascendc on A5, or torch_native. |

| Return value | Type | Description |
| --- | --- | --- |
| `y` | Tensor | FP8 with shape [..., C]. |
| `scale` | Tensor | MX returns E8M0 [..., C/64, 2]. Modes 1/3 return [..., C/128], with FP32 dtype except mode 3 with ue8m0_scale=True uses E8M0. |
| `y_origin` | Tensor | Input dtype and shape [..., C]. Defined only in mode 3 with output_origin=True. |

## ascendc backend

Runs fused activation and quantization on A5.

## torch_native backend

Uses PyTorch operations on the input devices.

## Example

Run on arch35 with the AscendC implementation installed.

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.activation import swiglu_group_quant

x = torch.randn(4, 512, device="npu", dtype=torch.bfloat16)
counts = torch.tensor([1, 0, 3], device="npu", dtype=torch.int64)
y, scale, _ = swiglu_group_quant(x, group_index=counts, quant_mode=2, impl="ascendc")
print(y.shape, scale.shape)
```

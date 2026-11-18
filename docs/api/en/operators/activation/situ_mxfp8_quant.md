# situ_mxfp8_quant

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Unsupported | Supported |
| `torch_native` | Unsupported | Supported |

The operator is A5-only: `SUPPORTED_ARCHS` is `3510`, so arch22 does not build
it, and the Python layer reports every implementation, including the PyTorch
reference, as unavailable on that generation. arch35 was verified on Ascend 950.

## Signature

```python
from sgl_kernel_npu.operators.activation import situ_mxfp8_quant

def situ_mxfp8_quant(
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 1,
    beta: float = 4.0,
    linear_beta: float = 25.0,
    *,
    impl="auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

`auto` and `ascendc` call `torch.ops.npu.situ_mxfp8_quant`; `torch_native` selects
the pure PyTorch reference. Backend modules are imported lazily. The native
operator is not registered in arch22 wheels or in subset wheels that omit
`situ_mxfp8_quant`, and there is no fallback: the Torch lookup fails unless
`impl="torch_native"` is requested.

## Description

Fuses the Kimi-K3 SiTU activation with per-32-element MXFP8 quantization. The
input is the BF16 GMM1 output `[capacity, 6144]`; columns `[0, 3072)` are the
gate branch and `[3072, 6144)` the up branch. [SGLang
NPUSituMXFP8Quant](https://github.com/sgl-project/sglang/blob/5481134403e01b46891e091ec63db42a36e6626f/python/sglang/srt/hardware_backend/npu/moe/activation.py#L163-L185)
calls this operator between the packed gate/up projection and the grouped down
projection.

$$
g = x[:, :3072], \qquad u = x[:, 3072:],
$$

$$
a = \beta\tanh(g / \beta)\,\sigma(g), \qquad
v = L\tanh(u / L), \qquad
y = a v, \qquad
L = \mathrm{linear\_beta}.
$$

All intermediates use float32; `y` is rounded to BF16 with round-to-nearest-even,
exactly as the unfused path would leave it, and then quantized to MXFP8.

Each 32 consecutive values of the 3072-column BF16 result share one E8M0
(power-of-two) scale derived from the group's maximum BF16 exponent. The scaled
values are saturated to ±448 and converted to E4M3FN with round-to-nearest-even;
the saturation is explicit, not merely implied by the format's finite maximum.
An all-zero group stores scale byte `0`; a group containing non-finite values
stores scale byte `255`.

Only the row count is taken from `group_list`: `sum(group_list)` for type `1` and
`group_list[-1]` for type `0`. Individual expert boundaries are not consumed, so
any list producing the same total behaves identically. The first
`min(valid_rows, capacity)` rows are written; the remaining rows are padding
left uninitialized, matching the downstream grouped matmul contract.

## Parameters and return values

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `hidden_states` | `torch.Tensor` | `[capacity, 6144]` | `torch.bfloat16` | Required | Contiguous tensor on an Ascend NPU with `capacity > 0`. Gate is the first half, up the second half. |
| `group_list` | `torch.Tensor` | `[E]` | `torch.int32`, `torch.int64` | Required | Contiguous, non-empty one-dimensional tensor on the same NPU. |
| `group_list_type` | `int` | — | — | `1` | `0` for cumulative per-expert counts (the last element is the valid row count), `1` for per-expert counts (their sum is the valid row count). Only that row count is consumed. |
| `beta` | `float` | — | — | `4.0` | Positive gate soft-saturation bound. |
| `linear_beta` | `float` | — | — | `25.0` | Positive up soft-saturation bound. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"ascendc"` or `"torch_native"`. |

The public operator checks the fixed width, dtype, contiguity, `group_list`
shape/dtype/device, `group_list_type` and the positive bounds before dispatch;
violations raise `ValueError`, except non-contiguous inputs, which raise
`RuntimeError`. Device-side group values are the caller's responsibility: they
are not validated, and totals above `capacity` are clamped to `capacity`.

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `payload` | `torch.Tensor` | `[capacity, 3072]` | `torch.float8_e4m3fn` | Newly allocated contiguous tensor on the input device. Only the valid prefix is defined. |
| `scales` | `torch.Tensor` | `[capacity, 48, 2]` | `torch.float8_e8m0fnu` | Vendor `npu_dynamic_mx_quant` layout: 96 contiguous E8M0 bytes per row, one per 32-element group, viewed as `[48, 2]`. |

## ascendc backend

- Single CANN kernel entry `sgl_situ_mxfp8_quant`, declared with `SUPPORTED_ARCHS 3510`. Tiling mirrors the legacy `blockDim = min(GetCoreNumAiv(), rows)` scheduling, and rows are distributed as `ceil_div(valid_rows, blockDim)` per core.
- The problem size is fixed at `[capacity, 6144] -> [capacity, 3072]`; there is no dynamic shape or column tiling.
- The Kernel reads `group_list` on device, reduces it to the valid row count, and processes only that prefix; per-expert partitions are not used.
- SiTU runs in fp32, is rounded to BF16, and is then quantized with the vendor micro-scaling primitives in `mx_quant_kernel.h`. E4M3FN uses round-to-nearest-even with explicit ±448 saturation.
- Padding rows are not written; their contents are undefined and must not be consumed.

## torch_native backend

- Pure PyTorch reference in `implementation/torch_native/activation/situ_mxfp8_quant.py`; it does not call the Kernel and can run on CPU.
- Reproduces the fp32 SiTU, the BF16 round trip, the per-32-element E8M0 scale construction (including `0` for all-zero groups and `255` for non-finite groups) and the E4M3FN cast with ±448 saturation.
- It does not model the Kernel's bitwise rounding differences, so compare dequantized results with a dtype-appropriate tolerance rather than bitwise. Intended for accuracy comparison, not performance.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.activation import situ_mxfp8_quant

capacity = 128
counts = torch.tensor([0, 7, 13, 0, 9, 3, 0, 5], dtype=torch.int64, device="npu")
x = (torch.randn(capacity, 6144, dtype=torch.float32) * 2).to(torch.bfloat16).npu()

payload, scales = situ_mxfp8_quant(x, counts, group_list_type=1)  # auto -> ascendc
assert payload.shape == (capacity, 3072)
assert scales.shape == (capacity, 48, 2)
```

## Limitations

- The problem size is fixed at `[capacity, 6144] -> [capacity, 3072]`.
- A5-only; arch22 wheels do not contain this operator and the Python layer rejects every implementation there.
- Device-side `group_list` values are not validated; totals above `capacity` are clamped to `capacity` instead of rejected.
- Padding rows are undefined by design.
- The `torch_native` reference is intended for accuracy comparison, not performance.

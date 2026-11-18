# dynamic_quant

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | Supported | Supported |
| `torch_native` | Supported | Supported |

The vendor `npu_dynamic_quant` op is a per-row symmetric INT8 dynamic
quantization operator, available on both arch22 (Ascend910) and arch35
(Ascend950).

## Signature

```python
from sgl_kernel_npu.operators.activation import dynamic_quant

def dynamic_quant(
    x: torch.Tensor,
    dst_type: torch.dtype = torch.int8,
    *,
    impl="auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

`auto` and `torch_npu` call `torch_npu.npu_dynamic_quant`; `torch_native`
selects the pure PyTorch reference. Backend modules are imported lazily.

## Description

Per-row symmetric INT8 dynamic quantization along the last dimension: each row
(last dimension) shares one scale `amax / 127`, where `amax` is the row's
maximum absolute value. The scaled values are rounded and saturated to ±127
before the INT8 cast. An all-zero row keeps scale `0` and a zero payload.

$$
s_r = \frac{\max_{i \in r} |x_i|}{127}, \qquad
y_i = \mathrm{INT8}\big(\mathrm{clamp}(\mathrm{round}(x_i / s_r), \pm 127)\big).
$$

A typical use is the GLM-5.3-Flash DSA indexer INT8 query quantization: reshape
`[tokens, heads, 128]` to `[-1, 128]` and call this operator to obtain the INT8
payload plus one FP32 scale per row for the downstream quantized indexer
computation.

## Parameters and return values

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[..., D]` with `D > 0` | `torch.bfloat16`, `torch.float16`, `torch.float32` | Required | Contiguous tensor on an Ascend NPU with at least one dimension. |
| `dst_type` | `torch.dtype` | — | — | `torch.int8` | Only `torch.int8` is supported. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"torch_npu"` or `"torch_native"`. |

The public operator checks the rank, the last-dimension width, dtype,
contiguity and `dst_type` before dispatch; violations raise `ValueError`,
except non-contiguous inputs, which raise `RuntimeError`. Quantization is
per-row along the last dimension; the axis is not exposed.

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `payload` | `torch.Tensor` | Same as `x` | `torch.int8` | Newly allocated contiguous tensor on the input device. |
| `scale` | `torch.Tensor` | `x.shape[:-1]` | `torch.float32` | One dequantization scale (`amax / 127`) per row; an all-zero row keeps scale `0`. |

## torch_npu backend

- Binds the vendor operator `torch_npu.npu_dynamic_quant` directly; no new Ascend C kernel is added.
- The input is reshaped to `[-1, D]` for the vendor call, then the payload is reshaped back to `x.shape` and the scale normalized to `torch.float32` with shape `x.shape[:-1]`.
- The quantization semantics (per-row `amax / 127` symmetric quantization, ±127 saturation) are guaranteed by the vendor operator.

## torch_native backend

- Pure PyTorch reference in `implementation/torch_native/activation/dynamic_quant.py`; it does not call the Kernel and can run on CPU.
- Reproduces the per-row `amax / 127` scale construction (including `0` for all-zero rows) and the INT8 cast with ±127 saturation, for any `[..., D]` shape and bf16/fp16/fp32 inputs.
- It does not model the Kernel's bitwise rounding differences, so compare dequantized results with a dtype-appropriate tolerance rather than bitwise. Intended for accuracy comparison, not performance.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.activation import dynamic_quant

x = torch.randn(4096, 128, dtype=torch.bfloat16, device="npu")
payload, scale = dynamic_quant(x)  # auto -> torch_npu
assert payload.shape == (4096, 128)
assert payload.dtype == torch.int8
assert scale.shape == (4096,)
assert scale.dtype == torch.float32
```

## Limitations

- Only `dst_type=torch.int8` is supported; quantization is per-row along the last dimension.
- The last dimension must be positive; the input must be contiguous.
- The `torch_native` reference is intended for accuracy comparison, not performance.

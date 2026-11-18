# dynamic_mx_quant

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | Unsupported | Supported |
| `torch_native` | Unsupported | Supported |

The operator is A5-only: the vendor `npu_dynamic_mx_quant` op is A5-specific,
and the Python layer reports every implementation, including the PyTorch
reference, as unavailable on arch22.

## Signature

```python
from sgl_kernel_npu.operators.activation import dynamic_mx_quant

def dynamic_mx_quant(
    x: torch.Tensor,
    dst_type: torch.dtype = torch.float8_e4m3fn,
    *,
    impl="auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

`auto` and `torch_npu` call `torch_npu.npu_dynamic_mx_quant` with `axis=-1`
and a fixed block size of 32; `torch_native` selects the pure PyTorch
reference. Backend modules are imported lazily. There is no fallback unless
`impl="torch_native"` is requested.

## Description

Dynamic MXFP8 quantization along the last dimension: each 32 consecutive
values share one E8M0 (power-of-two) scale derived from the floor(log2) of the
group's maximum absolute value, clamped so the largest scaled value fits the
E4M3FN range. The scaled values are saturated to ±448 and converted to E4M3FN
with round-to-nearest-even; the saturation is explicit, not merely implied by
the format's finite maximum. An all-zero group stores scale byte `0`; a group
containing non-finite values stores scale byte `255` and a NaN payload.

$$
e_g = \lfloor \log_2 \max_{i \in g} |x_i| \rfloor, \qquad
s_g = 2^{e_g - 8}, \qquad
y_i = \mathrm{E4M3FN}\big(\mathrm{clamp}(x_i / s_g, \pm 448)\big).
$$

The E8M0 scale byte is `e_g - 8 + 127`, clamped to a minimum of `0`.

A typical use is the GLM-5.3-Flash DSA indexer query quantization: reshape
`[tokens, heads, 128]` to `[-1, 128]` and call this operator to obtain the
E4M3FN payload plus one E8M0 scale per 32 elements for the downstream
quantized indexer computation.

## Parameters and return values

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[..., D]` with `D > 0` and `D % 32 == 0` | `torch.bfloat16`, `torch.float16`, `torch.float32` | Required | Contiguous tensor on an Ascend NPU with at least one dimension. |
| `dst_type` | `torch.dtype` | — | — | `torch.float8_e4m3fn` | Only `torch.float8_e4m3fn` is supported. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"torch_npu"` or `"torch_native"`. |

The public operator checks the rank, the last-dimension width, dtype,
contiguity and `dst_type` before dispatch; violations raise `ValueError`,
except non-contiguous inputs, which raise `RuntimeError`. `axis=-1` and the
block size of 32 are fixed and not exposed.

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `payload` | `torch.Tensor` | Same as `x` | `torch.float8_e4m3fn` | Newly allocated contiguous tensor on the input device. |
| `scales` | `torch.Tensor` | `x.shape[:-1] + (D // 32,)` | `torch.float8_e8m0fnu` | One E8M0 scale byte per 32-element group; a vendor uint8 or `[..., D // 64, 2]` layout is normalized to this shape. |

## torch_npu backend

- Binds the vendor operator `torch_npu.npu_dynamic_mx_quant` directly with `axis=-1` and the vendor default block size of 32; no new Ascend C kernel is added.
- The quantization semantics (E8M0 scale derivation, ±448 saturation, the `0`/`255` special bytes) are guaranteed by the vendor micro-scaling primitives.
- The vendor scale output is normalized to `torch.float8_e8m0fnu` with shape `[..., D // 32]`.
- There is no silent fallback to another implementation; on arch22 the generation registry reports the operator as unavailable.

## torch_native backend

- Pure PyTorch reference in `implementation/torch_native/activation/dynamic_mx_quant.py`; it does not call the Kernel and can run on CPU.
- Reproduces the per-32-element E8M0 scale construction (including `0` for all-zero groups and `255` for non-finite groups) and the E4M3FN cast with ±448 saturation, for any `[..., D]` shape and bf16/fp16/fp32 inputs.
- It does not model the Kernel's bitwise rounding differences, so compare dequantized results with a dtype-appropriate tolerance rather than bitwise. Intended for accuracy comparison, not performance.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.activation import dynamic_mx_quant

x = torch.randn(4096, 128, dtype=torch.bfloat16, device="npu")
payload, scales = dynamic_mx_quant(x)  # auto -> torch_npu
assert payload.shape == (4096, 128)
assert payload.dtype == torch.float8_e4m3fn
assert scales.shape == (4096, 4)
assert scales.dtype == torch.float8_e8m0fnu
```

## Limitations

- A5-only; every implementation is unavailable on arch22.
- Only `dst_type=torch.float8_e4m3fn` is supported; `axis=-1` and the block size of 32 are fixed.
- The last dimension must be a positive multiple of 32; the input must be contiguous.
- The `torch_native` reference is intended for accuracy comparison, not performance.

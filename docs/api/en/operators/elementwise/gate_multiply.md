# gate_multiply

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Supported |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

The fused kernel and its threshold were measured on arch22 (Ascend910_9362 / 910C). The Triton backend is portable but its benefit on other NPUs is unverified; below the threshold, and on arch35, it evaluates the same PyTorch expression as `torch_native`.

## Signature

```python
from sgl_kernel_npu.operators.elementwise import gate_multiply

def gate_multiply(y, gate, *, impl="auto"):
    ...
```

`auto` and `triton` are registered to the same entry point. The default uses Triton, with backend modules imported lazily. Unsupported inputs or unavailable implementations raise an error without automatic fallback.

## Description

GDN's sigmoid-after-normalization path promotes a BF16 normalized activation to FP32, multiplies it by a stored FP32 sigmoid gate, and casts the FP32 product back to BF16. This inference-only operator performs that multiply and cast:

$$
\mathrm{out} = \mathrm{cast}_{\mathrm{bf16}}\big(\mathrm{cast}_{\mathrm{fp32}}(y)\odot \mathrm{gate}\big).
$$

The caller computes `gate` with its existing sigmoid before this call. The BF16 normalization rounding already applied to `y` is preserved; normalization is not fused and sigmoid is not approximated. This operator is not a replacement for the swish-gated layernorm kernel and does not change model dispatch. No backward is provided.

## Parameters and return values

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `y` | `torch.Tensor` | any | `torch.bfloat16` | Required | BF16 normalized activation on an NPU. |
| `gate` | `torch.Tensor` | same as `y` | `torch.float32` | Required | FP32 sigmoid gate; must share `y`'s shape and NPU device. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"triton"`, or `"torch_native"`. |

Both inputs must be on the same NPU, must not require gradients, and must have matching shapes. Contract violations raise `ValueError` before any kernel launch.

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `out` | `torch.Tensor` | same as `y` | `torch.bfloat16` | Newly allocated result on `y`'s device. Inputs are not modified. |

## triton backend

- Preserves the computation of the original kernel. Loads `y` as BF16, promotes to FP32, multiplies by the FP32 `gate` and stores BF16.
- Contiguous inputs with at least 16,777,216 elements use the fused NPU kernel; smaller or strided inputs retain the PyTorch expression. The threshold was measured on 910C and is not a claim of benefit on every NPU.
- The vector-core count comes from the shared device-properties helper. The runtime element count does not specialize the kernel, so warm the shape and kernel before graph capture.
- The fused result is bitwise equal to the PyTorch expression: finite values match exactly and NaN locations coincide.

## torch_native backend

- Uses independent PyTorch operations without calling Triton: `(y.float() * gate).to(y.dtype)`.
- This is the accuracy reference. Because the Triton backend reproduces it exactly, the two backends are compared with zero tolerance (finite values are bitwise equal; NaN placement is compared separately).

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.elementwise import gate_multiply

# Keep the existing normalization and sigmoid implementations.
y = torch.randn(4096, 4096, device="npu", dtype=torch.bfloat16)  # normalized, BF16
z = torch.randn(4096, 4096, device="npu")
gate = torch.sigmoid(z.float())  # FP32 gate

out = gate_multiply(y, gate)
ref = gate_multiply(y, gate, impl="torch_native")
assert torch.equal(out, ref)
```

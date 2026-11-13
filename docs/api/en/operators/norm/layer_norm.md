# layer_norm

LayerNorm normalization operator.

## Hardware support

| Implementation | arch22 (2201) | arch35 (3510) |
| --- | --- | --- |
| `cann_ops_transformer` | Supported | Unsupported |
| `torch_native` | Supported | Unsupported |

## Signature

```python
from sgl_kernel_npu.operators.norm import layer_norm

def layer_norm(
    input,
    normalized_shape,
    weight=None,
    bias=None,
    eps=1e-5,
    impl_mode=0,
    impl="auto",
):
    ...
```

`auto` selects `cann_ops_transformer` on arch22. There is no fallback: requesting the operator
on arch35 raises instead of switching backends. The legacy name `layernorm` is exported as
an alias.

## Description

Normalizes the trailing `len(normalized_shape)` dimensions of `input`:

```text
out = (input - E[input]) / sqrt(Var[input] + eps) * weight + bias
```

Leading dimensions that are not normalized are treated as independent rows; `weight` and
`bias` are optional affine parameters.

## Parameters and return values

`A...` are the leading dimensions that are not normalized; `R...` are the normalized ones.

| Parameter | Shape | Dtype / default | Description |
| --- | --- | --- | --- |
| `input` | `[A..., R...]` | FP16 / BF16 / FP32 | At least one dimension, and no fewer than `normalized_shape`. |
| `normalized_shape` | `[R...]` | Required | Host integer list of trailing dimensions to normalize; must be non-empty. |
| `weight` | `[R...]` | `None` | Optional scale; dtype matches `input` or is FP32. `weight` and `bias` must share one dtype when both are present. |
| `bias` | `[R...]` | `None` | Optional offset with the same dtype rules as `weight`. |
| `eps` | — | float, `1e-5` | Added to the variance for numerical stability. |
| `impl_mode` | — | int, 0 | CANN computation mode, forwarded to ACLNN unchanged. |
| `impl` | — | `"auto"` | Keyword-only: `auto`, `cann_ops_transformer`, `torch_native`. |

| Return | Shape | Dtype |
| --- | --- | --- |
| `out` | Same as `input` | Same as `input` |
| `mean` | `[A..., 1, ...]` | Same as `input` |
| `rstd` | `[A..., 1, ...]` | Same as `input` |

`mean` and `rstd` keep the input rank and place ones on the normalized axes; without leading
axes the shape is all ones.

## cann_ops_transformer backend

- Calls the CANN toolkit operator `aclnnLayerNormWithImplMode`. Symbols resolve in the shared
  order: the wheel's `libcust_opapi.so` first, then the CANN toolkit `libopapi.so`. No vendor
  package or extra `ASCEND_CUSTOM_OPP_PATH` entry is needed.
- `out`, `mean` and `rstd` are allocated with the input options and NPU format before the
  launch, matching the legacy binding.
- `normalized_shape` and `eps` are host-side parameters; never pass NPU tensors for them.
- No extra validation, dtype conversion or contiguity copy is added around the ACLNN call;
  inputs must already satisfy the operator contract.
- arch22 (A2/A3) only.

## torch_native backend

- Pure PyTorch reference: computes in FP32 and casts `out`, `mean` and `rstd` back to the
  input dtype.
- `mean` and `rstd` are reshaped to the legacy `[A..., 1, ...]` allocation shape.
- `impl_mode` only selects a CANN computation mode; the reference ignores it.
- Registered for arch22 only, like the device implementation.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import layer_norm

values = torch.randn(512, 4096, device="npu", dtype=torch.float16)
weight = torch.ones(4096, device="npu", dtype=torch.float32)
bias = torch.zeros(4096, device="npu", dtype=torch.float32)

out, mean, rstd = layer_norm(values, [4096], weight, bias, eps=1e-5)
print(out.shape)   # [512, 4096]
print(mean.shape)  # [512, 1]
```

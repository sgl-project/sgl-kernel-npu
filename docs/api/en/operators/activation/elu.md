# elu

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Supported (measured on Ascend910 + CANN 9.1.0) | Pending verification (`3510` is a build declaration only; no measurement) |
| `torch_native` | Supported | Pending verification |

No `triton` implementation is declared: passing `impl="triton"` raises `ImplementationNotFoundError`.

## Signature

The default option is `auto`, which currently registers the same entry point as `ascendc`. Omitting `impl` or passing `"auto"` calls the Ascend C operator; `auto` does not fall back to another implementation when the native operator is missing.

```python
from sgl_kernel_npu.operators.activation import elu

def elu(x, alpha=1.0, *, impl="auto"):
    ...
```

## Description

Computes ELU (Exponential Linear Unit) element-wise, where $\alpha$ is the coefficient:

$$
\mathrm{elu}(x)=
\begin{cases}
x, & x > 0 \\
\alpha\left(e^{x}-1\right), & x \le 0
\end{cases}
$$

Instead of branching per element, the Ascend C implementation rewrites the piecewise function using $\max(x,0)=x-\min(x,0)$:

$$
\mathrm{elu}(x)=\alpha\left(e^{\min(x,0)}-1\right)+\left(x-\min(x,0)\right)
$$

`Exp` therefore only sees $\min(x,0)\le 0$, whose result stays in $(0,1]$, so fp16 cannot overflow. The `torch_native` reference evaluates `torch.expm1` in float32 (avoiding precision loss of $e^{x}-1$ near zero) and casts back to the input dtype.

Numerical note: in fp16 the kernel evaluates $e^{\min(x,0)}-1$ directly in fp16, so the *relative* error grows as $x\to0^-$ (~2.3% at $x=-0.001$). The *absolute* error there is only ~2.3e-5, which the `atol` term of the usual tolerance absorbs, so the standard `1e-3` still holds. Use fp32 when relative accuracy in that narrow band matters.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | Any shape, at least 0 elements | `torch.float16`, `torch.float32` | Required | Contiguous tensor on the NPU. The kernel processes the flattened element sequence, so no particular rank or alignment is required. |
| `alpha` | `float` | — | — | `1.0` | Coefficient of the negative branch; must be finite. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only. One of `"auto"`, `"ascendc"`, `"torch_native"`. `auto` currently selects the Ascend C implementation. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | Same as `x` | Same as `x` | New tensor on the same device; the input is not modified. |

Additional notes: the Ascend C implementation requires `x` to be contiguous with at least one dimension. Inputs whose element count is not a multiple of `blockDim × tileLength` are zero-padded on the host, computed, then trimmed back to the original length and shape; aligned activation tensors, the common case in LLMs, take the zero-overhead fast path. Unsupported dtypes are rejected before the kernel is launched.

## ascendc backend

- Implemented with Ascend C `op_host` / `op_kernel` and launched directly through `EXEC_KERNEL_CMD`; no workspace and no GE tiling.
- The host computes `tileLength` from the UB capacity and dtype (512-element aligned, capped at 4096) and picks `blockDim` from the tile count and AIV core count.
- Accepts contiguous inputs of any rank: flatten, pad when needed, compute, then trim and restore the original shape.

## torch_native backend

- Evaluates the formula with PyTorch ops in float32 and casts back to the input dtype.
- Accepts any shape and non-contiguous inputs; serves as the independent reference implementation.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.activation import elu

x = torch.randn(16, 7168, device="npu", dtype=torch.float16)

print(list_impls(elu))  # ['ascendc', 'auto', 'torch_native']
y_default = elu(x)  # default auto, which selects Ascend C
y_ascendc = elu(x, alpha=1.0, impl="ascendc")
y_native = elu(x, alpha=1.0, impl="torch_native")

torch.testing.assert_close(y_default, y_ascendc, rtol=0, atol=0)
torch.testing.assert_close(y_ascendc, y_native, rtol=1e-3, atol=1e-3)
```
